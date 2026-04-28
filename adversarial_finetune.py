import torch
import torch.nn as nn
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer,LlamaForCausalLM
from utils.transformers_config import TransformersConfig

# ==========================================
# [区域 1] 依赖引用
# ==========================================
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from evaluation.tools.text_editor import UPVGradientAttack
from watermark.upv.network_model import UPVDetector

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# [区域 2] 模型定义 (保持不变，遵照用户要求)
# ==========================================
class SubNet(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim=64):
        super(SubNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, bit_number, b_layers, input_dim, hidden_dim, num_classes=1, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.binary_classifier = SubNet(bit_number, b_layers)
        self.classifier = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x1 = x.view(batch_size*seq_len, -1)
        features = self.binary_classifier(x1)
        features = features.view(batch_size, seq_len, -1)
        output, _ = self.classifier(features)
        output = self.fc_hidden(output[:, -1, :])
        output = self.sigmoid(output)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output

# ==========================================
# [区域 3] 辅助函数
# ==========================================
def inputs_to_binary_tensor(input_ids, bit_number, device):
    mask = 2 ** torch.arange(bit_number - 1, -1, -1).to(device)
    return input_ids.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def sync_attacker(attack, model):
    """
    安全地将 model 的最新权重同步给 attack 内部的检测器。
    自动探测 attack 对象内部的模型属性名（model / detector / net 等）。
    """
    candidate_attrs = ['model', 'detector', 'net', 'classifier']
    for attr in candidate_attrs:
        if hasattr(attack, attr):
            internal_model = getattr(attack, attr)
            if isinstance(internal_model, nn.Module):
                try:
                    internal_model.load_state_dict(model.state_dict())
                    return attr  # 返回成功同步的属性名，供日志使用
                except RuntimeError as e:
                    print(f"[!] Sync via attack.{attr} failed: {e}")
    raise AttributeError(
        "[ERROR] Cannot find a valid nn.Module attribute in attack object. "
        "Please manually specify the attribute name in sync_attacker()."
    )


def evaluate(model, detector_tokenizer, texts, labels, bit_number, max_len, device):
    """
    在给定文本上快速评估模型准确率，用于训练中途监控效果。
    """
    model.eval()
    with torch.no_grad():
        tokens = detector_tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_len, return_tensors='pt'
        ).to(device)
        inputs_bin = inputs_to_binary_tensor(tokens['input_ids'], bit_number, device)
        outputs = model(inputs_bin).squeeze(1)
        preds = (outputs >= 0.5).float()
        labels_t = torch.tensor(labels, dtype=torch.float32).to(device)
        acc = (preds == labels_t).float().mean().item()
    model.train()
    return acc

# ==========================================
# [区域 4] 主程序
# ==========================================
def main():
    # --- 1. 配置参数 ---
    ALGORITHM_NAME = 'UPV'
    DETECTOR_PATH = './model/finetuned_robust/duikangxunlian5000opt-1.3b.pt'
    OUTPUT_DIR = './model/finetuned_robust/'

    # 【修复】使用更小的学习率 + 梯度裁剪，防止双重 sigmoid 结构下梯度爆炸
    LR = 1e-4
    GRAD_CLIP = 1.0

    BATCH_SIZE = 40
    BIT_NUMBER = 16
    MAX_LEN = 200
    TOTAL_SAMPLES = 5000   # 正式训练建议 5000+，快速验证可改为 500

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. 初始化资源 ---
    print("[*] Loading Resources...")
    my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    detector_tokenizer = AutoTokenizer.from_pretrained("./huggingface/opt-1.3b")
    #detector_tokenizer = LlamaTokenizer.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf")
    detector_tokenizer.pad_token = detector_tokenizer.eos_token

    transformers_config = TransformersConfig(
        model=AutoModelForCausalLM.from_pretrained("./huggingface/opt-1.3b").to(device),
        #model=LlamaForCausalLM.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf").to(device),
        tokenizer=detector_tokenizer,
        vocab_size=502027, device=device, max_new_tokens=200, min_length=230,
        do_sample=True, no_repeat_ngram_size=4
    )
    my_watermark = AutoWatermark.load(
        ALGORITHM_NAME,
        algorithm_config=f'config/{ALGORITHM_NAME}.json',
        transformers_config=transformers_config
    )

    print("[*] Initializing Attacker...")
    attack = UPVGradientAttack(
        model_path=DETECTOR_PATH,
        tokenizer_name='./huggingface/opt-1.3b',
        #tokenizer_name='/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf',
        device=device
    )

    # --- 3. 初始化受害者模型 ---
    print("[*] Loading Victim Model...")
    model = TransformerClassifier(bit_number=16, b_layers=5, input_dim=64, hidden_dim=128)
    model.load_state_dict(torch.load(DETECTOR_PATH), strict=False)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # 【新增】学习率调度：每 200 步衰减一次，避免后期震荡
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    loss_fn = torch.nn.BCELoss()

    # --- 4. 训练前首次同步 ---
    print("[*] Syncing initial weights to attacker...")
    synced_attr = sync_attacker(attack, model)
    print(f"    -> Synced via attack.{synced_attr}")

    # --- 5. 动态对抗训练循环 ---
    print(f"\n[*] Starting Adversarial Training ({TOTAL_SAMPLES} samples, "
          f"update every {BATCH_SIZE})...")

    buffer_texts  = []
    buffer_labels = []
    global_step   = 0

    # 用于评估的小型固定集（每 10 步评估一次）
    eval_texts  = []
    eval_labels = []

    for i in tqdm(range(1, TOTAL_SAMPLES )):
        try:
            prompt = my_dataset.get_prompt(i)

            # =====================================================
            # 【核心修复】标签逻辑
            #
            # 目标：训练检测器识别"攻击后仍残留水印的文本"
            #
            #  水印文本原文          → 标签 1.0（有水印）
            #  梯度攻击后的文本      → 标签 1.0（仍然有水印，检测器要识别出来）
            #  自然文本（无水印）    → 标签 0.0
            #
            # 三类样本同时训练，让检测器学会：
            #   a. 正常水印文本   → 检测为 1
            #   b. 被攻击水印文本 → 仍检测为 1（鲁棒性目标）
            #   c. 自然文本       → 检测为 0（保持低误报）
            # =====================================================

            # 生成水印原文
            watermark_text = my_watermark.generate_watermarked_text(prompt)
            # 对水印文本发起梯度攻击
            attack_text    = attack.edit(watermark_text)
            # 无水印的自然文本
            natural_text   = my_dataset.get_natural_text(i)

            buffer_texts.extend([watermark_text, attack_text, natural_text])
            buffer_labels.extend([1.0,           1.0,         0.0])

            # 同步积累少量样本用于评估（只收集前 200 条）
            if len(eval_texts) < 200:
                eval_texts.extend([watermark_text, attack_text, natural_text])
                eval_labels.extend([1.0, 1.0, 0.0])

            # === 凑够 Batch Size，执行训练与同步 ===
            if len(buffer_texts) >= BATCH_SIZE:

                # 1. Tokenize & 转二进制张量
                tokens = detector_tokenizer(
                    buffer_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors='pt'
                ).to(device)

                inputs_bin    = inputs_to_binary_tensor(tokens['input_ids'], BIT_NUMBER, device)
                labels_tensor = torch.tensor(
                    buffer_labels, dtype=torch.float32
                ).to(device).unsqueeze(1)

                # 2. 前向 + 反向
                optimizer.zero_grad()
                outputs = model(inputs_bin)
                loss    = loss_fn(outputs, labels_tensor)
                loss.backward()

                # 【新增】梯度裁剪，防止双重 sigmoid 结构引发的梯度异常
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                optimizer.step()
                scheduler.step()

                # 3. 同步最新权重给攻击器（让攻击者随模型同步进化）
                sync_attacker(attack, model)

                # 4. 监控输出
                if global_step % 10 == 0:
                    acc = evaluate(
                        model, detector_tokenizer,
                        eval_texts, eval_labels,
                        BIT_NUMBER, MAX_LEN, device
                    )
                    print(f"\n--- Step {global_step:4d} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Eval Acc: {acc*100:.1f}% | "
                          f"LR: {scheduler.get_last_lr()[0]:.2e} ---")

                # 5. 定期保存检查点
                if global_step % 50 == 0 and global_step > 0:
                    ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_step{global_step}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"    [Saved] {ckpt_path}")

                # 6. 清空缓冲区
                buffer_texts  = []
                buffer_labels = []
                global_step  += 1

        except Exception as e:
            print(f"\n[!] Error at index {i}: {e}")
            continue

    # --- 6. 保存最终模型 ---
    final_path = os.path.join(OUTPUT_DIR, "duikangxunlian5000opt-1.3b1.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[*] Done. Final model saved to: {final_path}")


if __name__ == '__main__':
    main()
import os
# 这一行必须在 import tensorflow 或 import OpenAttack 之前设置
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import OpenAttack as oa
import torch
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.tokenize.util import align_tokens
from evaluation.tools.text_editor import TextEditor
from watermark.upv.network_model import UPVGenerator, UPVDetector
import argparse
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
GLOBAL_ATTACK_LOGS = []


class UPVDetectorForOpenAttack(oa.Classifier):
    """将UPV检测器包装成OpenAttack兼容的分类器"""
    
    def __init__(self, detector, tokenizer, vocab_size, bit_number=16, device='cuda'):
        self.detector = detector
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.bit_number = bit_number
        self.device = device
    
    def get_pred(self, input_):
        """返回预测类别 (0=无水印, 1=有水印)"""
        return [1 if self.get_prob([text])[0][1] > 0.5 else 0 for text in input_]
    
    def get_prob(self, input_):
        """返回预测概率 [[p(无水印), p(有水印)], ...]"""
        results = []
        for text in input_:
            try:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                if len(token_ids) == 0:
                    results.append([0.5, 0.5])
                    continue
                    
                binary_input = self._tokens_to_binary_batch(token_ids).to(self.device)
                with torch.no_grad():
                    score = self.detector(binary_input)
                
                prob_watermark = score.item()
                prob_no_watermark = 1 - prob_watermark
                results.append([prob_no_watermark, prob_watermark])
            except Exception as e:
                print(f"Error processing text: {e}")
                results.append([0.5, 0.5])
        
        return np.array(results)
    
    def _token_to_binary(self, token_id):
        binary = format(token_id, f'0{self.bit_number}b')
        return torch.tensor([int(b) for b in binary], dtype=torch.float32)
    
    def _tokens_to_binary_batch(self, token_ids):
        binary_list = [self._token_to_binary(tid) for tid in token_ids]
        binary_tensor = torch.stack(binary_list)
        return binary_tensor.unsqueeze(0)


class OpenAttackWatermarkRemoval(TextEditor):
    """使用OpenAttack工具移除水印"""
    
    def __init__(self, detector, tokenizer, vocab_size, attack_method='textfooler', device='cuda',sem_model_name: str = "./huggingface/sbert_model"):
       
        super().__init__()
        
        # 包装检测器
        self.classifier = UPVDetectorForOpenAttack(detector, tokenizer, vocab_size, device=device)
        
        
        self.sem_model = SentenceTransformer(sem_model_name).to(device)
        self.device = device
        
        # 选择攻击器
        if attack_method == 'textfooler':
            self.attacker = oa.attackers.TextFoolerAttacker()#无效 
        elif attack_method == 'uat':
            self.attacker = oa.attackers.UATAttacker()#无效
        elif attack_method == 'bae':
            self.attacker = oa.attackers.BAEAttacker()#无效 
        elif attack_method == 'bertattack':
            self.attacker = oa.attackers.BERTAttacker()#8/10 全中
        elif attack_method == 'fd':
            self.attacker = oa.attackers.FDAttacker()#无效
        elif attack_method == 'pwws':
            #self.attacker = oa.attackers.PWWSAttacker(tokenizer=SpacePreservingTokenizer())#全中 全中
            self.attacker = oa.attackers.PWWSAttacker()
        elif attack_method == 'genetic':
            self.attacker = oa.attackers.GeneticAttacker()#无效 全中
        elif attack_method == 'deepwordbug':
            self.attacker = oa.attackers.DeepWordBugAttacker()#无效 7/10
        elif attack_method == 'textbugger':
            self.attacker = oa.attackers.TextBuggerAttacker()#全中 全中
        elif attack_method == 'scpn':
            torch.serialization.add_safe_globals([argparse.Namespace])
            self.attacker = oa.attackers.SCPNAttacker()#全中
        elif attack_method == 'viper':
            self.attacker = oa.attackers.VIPERAttacker()#无效
        elif attack_method == 'gan':
            self.attacker = oa.attackers.GANAttacker()#无效
        elif attack_method == 'hotflip':
            self.attacker = oa.attackers.HotFlipAttacker()  # HotFlip不支持修改率参数
       
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
    
    def compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
       
        import torch.nn.functional as F

        embeddings = self.sem_model.encode(
            [text_a, text_b],
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True,   # L2 归一化，点积即余弦相似度
        )
        similarity = F.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0),
        ).item()
        return similarity
    
    def edit(self, text: str, reference=None):
        """使用OpenAttack移除水印"""
        global GLOBAL_ATTACK_LOGS
        # 获取原始分数
        original_pred = self.classifier.get_prob([text])[0]
        original_score = original_pred[1]  # 水印概率
    
        print(f"Original score: {original_score:.4f}")
    
        # 如果已经是无水印，直接返回
        if original_score < 0.5:
            return text
    
        # 执行攻击 (目标是将类别从1变为0)
        try:
            attack_eval = oa.AttackEval(self.attacker, self.classifier, 
                metrics=[
                
                oa.metric.Fluency(),
                #oa.metric.SemanticSimilarity(),
                oa.metric.EditDistance(),
                oa.metric.ModificationRate(),
                # oa.metric.GrammaticalErrors()
                ]
            )
            
            # 构建输入样本，注意 y 应为目标标签（我们希望模型将其分类为 0）
            input_data = [{"x": text, "y": 1}]  # 因为我们想把它从 1 改成 0
            result = attack_eval.eval(input_data, visualize=True, progress_bar=False)
            
            # 安全地尝试提取 best adversarial example
            
            first_adv = None
            for result1 in attack_eval.ieval(input_data):
                if result1["success"]:
                    first_adv = result1["result"]
                    sem_sim = self.compute_semantic_similarity(text, first_adv)
                    current_log = {"queries": result["Avg. Victim Model Queries"],
                                   "rate": result["Avg. Word Modif. Rate"],  # 对应 Modification 类里的 NAME 
                                   "sem":sem_sim,
                                    }
                    GLOBAL_ATTACK_LOGS.append(current_log)
                    return first_adv
                else:
                    return text
        except Exception as e:
            print(f"Error during attack: {e}")
            return text
            
           
    def print_final_statistics():
        global GLOBAL_ATTACK_LOGS
    
        if not GLOBAL_ATTACK_LOGS:
            print("[Stats] 没有记录到任何攻击数据。")
            return

    # 1. 过滤出攻击成功的样本 (通常只统计成功的 Query，或者全部统计，看你需要)
    # OpenAttack 的 eval 返回的已经是单个样本结果，如果失败通常 Queries 也会很高
        successful_logs = [log for log in GLOBAL_ATTACK_LOGS if log.get("queries", 0) > 0] 
    
        total_samples = len(GLOBAL_ATTACK_LOGS)
    
        if total_samples == 0: return

    # 2. 计算平均 Queries
        total_queries = sum(log["queries"] for log in GLOBAL_ATTACK_LOGS)
        avg_queries = total_queries / total_samples
    
    # 3. 计算平均修改率
        total_rate = sum(log["rate"] for log in GLOBAL_ATTACK_LOGS)
        avg_rate = total_rate / total_samples

        print("="*40)
        print(f"FINAL ATTACK STATISTICS (Total Samples: {total_samples})")
        print("-" * 40)
        print(f"Average Queries:      {avg_queries:.2f}")
        print(f"Average Modif Rate:   {avg_rate * 100:.2f}%") # 如果 rate 是 0.05 这种小数
        print("="*40)

# 在你跑完所有测试后调用它
# print_final_statistics()
     
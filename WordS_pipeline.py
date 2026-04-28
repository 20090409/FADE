
import torch
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.success_rate_calculator import FundamentalSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer,LlamaForCausalLM
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from evaluation.tools.text_editor import TruncatePromptTextEditor, SynonymSubstitution
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def assess_robustness():
    algorithm_name = 'UPV'
    #my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    my_dataset = C4Dataset('dataset/c4/final_processed_dbpedia.json')
    # 保持模型配置不变
    transformers_config = TransformersConfig(
        model=AutoModelForCausalLM.from_pretrained("./huggingface/gpt2").to(device),
        tokenizer=AutoTokenizer.from_pretrained("./huggingface/gpt2"),
        # model=LlamaForCausalLM.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf").to(device),
        # tokenizer=LlamaTokenizer.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf"),
        vocab_size=50272,
        device=device,
        max_new_tokens=200,
        min_length=230,
        do_sample=True,
        no_repeat_ngram_size=4
    )
    
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)

    # 1. TNR/FPR 通常不随攻击强度（ratio）变化，可以只在循环外计算一次
    pipeline_unwatermarked = UnWatermarkedTextDetectionPipeline(
        dataset=my_dataset, 
        text_editor_list=[],
        show_progress=True, 
        return_type=DetectionPipelineReturnType.IS_WATERMARKED
    )
    unwatermarked_results = pipeline_unwatermarked.evaluate(my_watermark)

    # 2. 定义 ratio 列表进行循环
    ratios = [0.4,0.5,0.6]
    calculator = FundamentalSuccessRateCalculator(labels=['TPR', 'FPR', 'TNR', 'FNR', 'F1'])

    print(f"{'Ratio':<10} | {'Metrics'}")
    print("-" * 50)

    for ratio in ratios:
        print(f"\nTesting robustness with ratio: {ratio}")
        
        # 在每一轮循环中，使用当前的 ratio 实例化 SynonymSubstitution
        pipeline_watermarked = WatermarkedTextDetectionPipeline(
            dataset=my_dataset, 
            text_editor_list=[TruncatePromptTextEditor(), SynonymSubstitution(ratio=ratio)],                             
            show_progress=True, 
            return_type=DetectionPipelineReturnType.IS_WATERMARKED
        ) 
        
        # 执行评估
        watermarked_results = pipeline_watermarked.evaluate(my_watermark)
        
        # 计算并打印结果
        results = calculator.calculate(watermarked_results, unwatermarked_results)
        print(f"Result for ratio {ratio}: {results}")

if __name__ == '__main__':

    assess_robustness()
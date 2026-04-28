
import torch
import time
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM,LlamaTokenizer
from evaluation.tools.text_editor import TruncatePromptTextEditor
from evaluation.tools.success_rate_calculator import FundamentalSuccessRateCalculator
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def assess_detectability(algorithm_name, labels, rules, target_fpr):
    my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    # transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("./huggingface/opt-1.3b").to(device),
    #                                          tokenizer=AutoTokenizer.from_pretrained("./huggingface/opt-1.3b"),
    transformers_config = TransformersConfig(model=LlamaForCausalLM.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf").to(device),
                                             tokenizer=LlamaTokenizer.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf"),
                                             vocab_size=50272,
                                             device=device,
                                             max_new_tokens=200,
                                             min_length=230,
                                             do_sample=True,
                                             no_repeat_ngram_size=4)
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)
    pipline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor()],
                                                show_progress=True, return_type=DetectionPipelineReturnType.IS_WATERMARKED) 

    pipline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.IS_WATERMARKED)
   
    calculator = FundamentalSuccessRateCalculator(labels=labels)
    
    print(calculator.calculate(pipline1.evaluate(my_watermark), pipline2.evaluate(my_watermark)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='UPV')
    parser.add_argument('--labels', nargs='+', default=['TPR', 'FPR','TNR','FNR','F1'])
    parser.add_argument('--rules', type=str, default='target_fpr')
    args = parser.parse_args()
    assess_detectability(args.algorithm, args.labels, args.rules, args.target_fpr)
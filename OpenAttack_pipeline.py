# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ================================================================
# assess_robustness.py
# Description: Assess the robustness of a watermarking algorithm
# ================================================================

import torch
import os
from openai import OpenAI
from translate import Translator
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator,FundamentalSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM,AutoModelForSeq2SeqLM,LlamaTokenizer,LlamaForCausalLM
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, BackTranslationTextEditor, UPVGradientAttack,RandomWalkAttack
from evaluation.tools.text_quality_analyzer import PPLCalculator, LogDiversityAnalyzer, BLEUCalculator, PassOrNotJudger, GPTTextDiscriminator
import time
import OpenAttack
from evaluation.tools.text_editor_attackmodel import OpenAttackWatermarkRemoval
from evaluation.pipelines.quality_analysis import DirectTextQualityAnalysisPipeline, QualityPipelineReturnType, ReferencedTextQualityAnalysisPipeline, ExternalDiscriminatorTextQualityAnalysisPipeline
from watermark.upv.network_model import UPVDetector
from evaluation.tools.oracle import QualityOracle
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# device='cpu'
def assess_robustness():
    algorithm_name ='UPV'
    my_dataset = C4Dataset('dataset/c4/final_processed_dbpedia.json')
   #  transformers_config = TransformersConfig(model=LlamaForCausalLM.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf").to(device),
   #                                           tokenizer=LlamaTokenizer.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf"),
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("./huggingface/opt-1.3b").to(device),
                                             tokenizer=AutoTokenizer.from_pretrained("./huggingface/opt-1.3b"),
                                             vocab_size=50272,
                                             device=device,
                                             max_new_tokens=200,
                                             min_length=230,
                                             do_sample=True,
                                             no_repeat_ngram_size=4)
    
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)

    detector = UPVDetector(bit_number=16, b_layers=5, input_dim=64, hidden_dim=128)
    #detector.load_state_dict(torch.load('./detectors/detector_llama_dbpedia_topk_w5_1.pt',map_location='cpu'),strict=False)
    detector.load_state_dict(torch.load('./detectors/detector_gpt2_dbpedia_topk_w5_1.pt'),strict=False)
    attack_methods = ['textfooler','genetic','pwws','bertattack','hotflip']
    

    pipline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.IS_WATERMARKED)
    calculator = FundamentalSuccessRateCalculator(labels=['TPR', 'FPR','TNR','FNR','F1'])
    for attack_method in attack_methods:
       print(f"attack_method={attack_method}")
       attack = OpenAttackWatermarkRemoval(detector=detector,
                                             tokenizer=AutoTokenizer.from_pretrained('./huggingface/opt-1.3b'),
                                             #tokenizer=LlamaTokenizer.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf"),
                                             vocab_size=50272,
                                             attack_method=attack_method,
                                             device='cpu')
       pipline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor(), attack],
                                                show_progress=True, return_type=DetectionPipelineReturnType.IS_WATERMARKED) 
    
    #calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'FPR','TNR','FNR','F1'], rule='target_fpr', target_fpr=0.01)
       
       print(calculator.calculate(pipline1.evaluate(my_watermark), pipline2.evaluate(my_watermark)))
  
       from evaluation.tools.text_editor_attackmodel import GLOBAL_ATTACK_LOGS
       total = len(GLOBAL_ATTACK_LOGS)
       total_queries = sum(log["queries"] for log in GLOBAL_ATTACK_LOGS)
       avg_queries = total_queries / total
    
    # 3. 计算平均修改率
       total_rate = sum(log["rate"] for log in GLOBAL_ATTACK_LOGS)
       avg_rate = total_rate / total
       
       avg_sem = sum(log["sem"] for log in GLOBAL_ATTACK_LOGS) / total
       print(f"FINAL ATTACK STATISTICS (Total Samples: {total})")
       print(f"Avg Queries (Succ):      {avg_queries:.2f}")
       print(f"Avg Rate:   {avg_rate * 100:.2f}%") # 如果 rate 是 0.05 这种小数
       print(f"Avg sem:   {avg_sem:.2f}")
       GLOBAL_ATTACK_LOGS.clear()
    
   
    

if __name__ == '__main__':
    
    assess_robustness()
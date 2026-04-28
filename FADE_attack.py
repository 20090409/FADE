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
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.success_rate_calculator import FundamentalSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer,LlamaForCausalLM
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from evaluation.tools.text_editor import  SynonymSubstitution, UPVGradientAttack

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def assess_robustness():
    algorithm_name = 'UPV'
    my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    # transformers_config = TransformersConfig(model=LlamaForCausalLM.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf").to(device),
    #                                            tokenizer=LlamaTokenizer.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf"),
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("./huggingface/gpt2").to(device),
                                             tokenizer=AutoTokenizer.from_pretrained("./huggingface/gpt2"),
                                             vocab_size=50272,
                                             device=device,
                                             max_new_tokens=200,
                                             min_length=230,
                                             do_sample=True,
                                             no_repeat_ngram_size=4)
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)
  

    attack = UPVGradientAttack(
            model_path='./detectors/detector_gpt2_c4_topk_w5_1.pt',
            #model_path='./detectors/detector_llama_c4_topk_w5_1.pt',
            tokenizer_name='./huggingface/gpt2',
            device=device,
            speed_mode="balanced",
            )

    # Unify attack success criterion with the final UPV detector in config/UPV.json.
    attack.set_success_verifier(
        lambda edited_text: not my_watermark.detect_watermark(edited_text, return_dict=True)["is_watermarked"]
    )
        
   
    
    pipline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor(), attack],
                                                show_progress=True, return_type=DetectionPipelineReturnType.IS_WATERMARKED) 

    pipline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.IS_WATERMARKED)
    
  
    calculator = FundamentalSuccessRateCalculator(labels=['TPR', 'FPR','TNR','FNR','F1'])
    wm_detect_results = pipline1.evaluate(my_watermark)
    nwm_detect_results = pipline2.evaluate(my_watermark)
    metrics = calculator.calculate(wm_detect_results, nwm_detect_results)
    print(metrics)
  

  
    from evaluation.tools.text_editor import GLOBAL_ATTACK_LOGS
        
    total = len(GLOBAL_ATTACK_LOGS)
    if total > 0:
        success_logs = [x for x in GLOBAL_ATTACK_LOGS if x['success']]
        succ_count = len(success_logs)
        avg_queries_succ = sum(x['queries'] for x in success_logs) / succ_count if succ_count else 0
        avg_total_forwards_succ = sum(x.get('total_queries', x['queries']) for x in success_logs) / succ_count if succ_count else 0
        avg_rate_succ = sum(x['rate'] for x in success_logs) / succ_count if succ_count else 0
        avg_sem_succ = sum(x['semantic_similarity'] for x in success_logs) / succ_count if succ_count else 0
        final_asr = metrics.get('FNR', 0.0) * 100.0
        print(f"Total: {total}")
        print(f"Success Samples: {succ_count}")
        print(f"ASR (Final Detector): {final_asr:.2f}%")
        print(f"ASR (Attack Log): {succ_count/total*100:.2f}%")
        print(f"Avg Search Queries (Succ): {avg_queries_succ:.1f}")
        print(f"Avg Total Forwards (Succ): {avg_total_forwards_succ:.1f}")
        print(f"Avg Rate (Succ): {avg_rate_succ:.2f}%")
        print(f"Avg Sem. (Succ): {avg_sem_succ:.2f}")
        GLOBAL_ATTACK_LOGS.clear()
    

    
 
if __name__ == '__main__':
    
    assess_robustness()

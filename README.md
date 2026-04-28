# FADE

FADE is a text watermark attack and evaluation project built upon [THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM).

## Setup

Install the main dependencies:

```bash
pip install torch transformers sentence-transformers nltk numpy tqdm translate psutil
```

Install optional dependencies if you want to run OpenAttack baselines or text quality evaluation:

```bash
pip install OpenAttack openai tiktoken sacrebleu bert-score rouge-score
```

Download NLTK resources:

```bash
python -m nltk.downloader wordnet omw-1.4 punkt averaged_perceptron_tagger
```

## Prepare Files

Before running, prepare the required local files and update paths in the scripts if needed:

- Dataset: `dataset/c4/processed_c4.json`
- Language model: `./huggingface/gpt2` or `./huggingface/opt-1.3b`
- Detector checkpoint: `./detectors/detector_gpt2_c4_topk_w5_1.pt`
- UPV config: `config/UPV.json`

Expected dataset format:

```json
{"prompt": "input prompt", "natural_text": "natural text"}
```

Also check `config/UPV.json` and make sure these paths are valid:

```json
"generator_model_name": "...",
"detector_model_name": "..."
```

## Run

Run the no-attack baseline:

```bash
python No_attack.py --algorithm UPV --labels TPR FPR TNR FNR F1
```

Run the FADE attack:

```bash
python FADE_attack.py
```

Run the synonym substitution baseline:

```bash
python WordS_pipeline.py
```

Run OpenAttack baselines:

```bash
python OpenAttack_pipeline.py
```

Run adversarial fine-tuning:

```bash
python adversarial_finetune.py
```

## Notes

- If CUDA device errors occur, edit the `device` variable in the corresponding script.
- If files are missing, update dataset, model, and checkpoint paths before running.

## Acknowledgements

This codebase is modified from [THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM). We thank the MarkLLM project for providing the original watermarking framework.


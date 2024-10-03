# VLEU: a Method for Automatic Evaluation for Generalizability of Text-to-Image Models

This repository contains the code and data for the paper "VLEU: a Method for Automatic Evaluation for Generalizability of Text-to-Image Models" presented at EMNLP 2024.

## Files

- `code/`: Contains the Python scripts and notebooks used for the project.
- `data/`: Contains the datasets used for training and evaluation.

## Quick Start

### Sampling Prompts

#### 1. GPT

`code/Visual Text Sampling/sample_prompts_gpt.py`
```sh
python sample_prompts_gpt.py \
  --api_key YOUR_KEY \
  --n_prompts 1000 \
  --output prompts.json \
  --n_threads 50 \
  --step 30 \
  --key_word dog
```

#### 2. LLaMA

`code/Visual Text Sampling/sample_prompts_llama.py`
```sh
python sample_prompts_llama.py \
  --model meta-llama/Llama-2-13b-chat-hf \
  --n_prompts 1000 \
  --output prompts.json \
  --num_return_sequences 2
```

### T2I Generation

#### 1. Stable Diffusion

`code/Visual Text Sampling/text2img_sd.py`
```sh
python text2img_sd.py \
  --model stabilityai/stable-diffusion-2-1 \
  --prompt_json_path prompts.json \
  --output_dir image_output \
  --num 1000 \
  --batch_size 4
```

#### 2. Stable Diffusion XL

`code/Visual Text Sampling/text2img_sdxl.py`
```sh
python text2img_sdxl.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt_json_path prompts.json \
  --output_dir image_output \
  --num 1000 \
  --batch_size 4
```

### VLEU Calculation

#### 1. CLIP

`code/VLEU Calculation/cal_vleu_clip.py`
```sh
python cal_vleu_clip.py \
  --model openai/clip-vit-base-patch16 \
  --prompt_json_path prompts.json \
  --image_dir image_output
```

#### 2. OpenCLIP

`code/VLEU Calculation/cal_vleu_openclip.py`
```sh
python cal_vleu_openclip.py \
  --model ViT-L-14 \
  --pretrained open_clip_pytorch_model.bin \
  --prompt_json_path prompts.json \
  --image_dir image_output
```

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@misc{cao2024vleumethodautomaticevaluation,
      title={VLEU: a Method for Automatic Evaluation for Generalizability of Text-to-Image Models}, 
      author={Jingtao Cao and Zheng Zhang and Hongru Wang and Kam-Fai Wong},
      year={2024},
      eprint={2409.14704},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.14704}, 
}
```
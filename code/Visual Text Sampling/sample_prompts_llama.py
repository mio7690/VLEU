import json
import argparse
import torch
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm


def get_prompt(prompt, num_return_sequences):
    prompt_len = len(prompt)
    sequences = pipeline(
        prompt,
        do_sample=True,
        temperature=0.3,
        top_k=10,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2048
    )
    responses = [seq['generated_text'][prompt_len:].strip() for seq in sequences]
    return responses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--n_prompts', type=int)
    parser.add_argument('--key_word', type=str, default=None)
    parser.add_argument('--output', type=str)
    parser.add_argument('--num_return_sequences', type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    system_prompt = ''

    if args.key_word:
        prompt = f'<s>[INST] <<SYS>>\n\n<</SYS>>\n\nPlease imagine a random picture and describe it in one sentence. [/INST] A lone wolf stands proudly atop a snow-covered mountain peak, its piercing gaze reflecting both strength and solitude. </s><s> [INST] Again [/INST]'
        response = 'A lone tree stands tall in a vast, snowy landscape, its branches adorned with delicate icicles glistening in the winter sunlight.'
    else:
        prompt = f'<s>[INST] <<SYS>>\n\n<</SYS>>\n\nPlease imagine a random picture and describe it in one sentence:'
        response = 'A lone tree stands tall in a vast, snowy landscape, its branches adorned with delicate icicles glistening in the winter sunlight.'
    
    prompts = []
    prompt_stack = []

    user_msg = 'Again'
    prompt += f' {response} </s><s> [INST] {user_msg} [/INST]'
    prompt_stack.append(prompt)

    n_pad_turn = 3
    n_pad = args.num_return_sequences ** (n_pad_turn + 1) - args.num_return_sequences
    n_prompts = args.n_prompts + n_pad

    progress_bar = tqdm(total=n_prompts)
    while len(prompts) < n_prompts:
        prompt = prompt_stack.pop(0)
        responses = get_prompt(prompt, args.num_return_sequences)
        prompts.extend(responses)
        prompt_stack.extend([f'{prompt} {resp} </s><s> [INST] {user_msg} [/INST]' for resp in responses])
        progress_bar.update(len(responses))
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(prompts[n_pad:n_prompts], f)

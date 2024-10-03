from diffusers import DiffusionPipeline
import argparse
import torch
import json, os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--prompt_json_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--num', type=int)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to("cuda")

    if args.prompt_json_path:
        prompts = json.load(open(args.prompt_json_path,'r',encoding='utf-8'))

    os.makedirs(args.output_dir, exist_ok=True)

    end = args.start + args.num
    for i in range(args.start, end, args.batch_size):
        size = min(args.batch_size, end - i)
        if args.prompt_json_path:
            images = pipe(prompts[i:i+size]).images
        else:
            images = pipe([args.prompt] * size).images
        for j, image in enumerate(images):
            image.save(f'{args.output_dir}/{i+j}.jpg')

from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
import os


def get_score(pipe, model, processor, prompts, batch_size=1, output_dir=None, temperature=0.01, device='cuda'):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)

    with torch.no_grad():
        text_embs = []
        img_embs = []
        for prompt in prompts:
            inputs = processor([prompt], return_tensors='pt', truncation=True)
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            outputs = model.get_text_features(**inputs)
            outputs /= outputs.norm(dim=-1, keepdim=True)
            text_embs.append(outputs)
        
        
        for i in range(0, len(prompts), batch_size):
            with torch.autocast("cuda"):
                images = pipe(prompts[i:i+batch_size]).images
            for j, image in enumerate(images):
                if output_dir:
                    image.save(f'{output_dir}/{i+j}.jpg')
                inputs = processor(images=image, return_tensors='pt')
                inputs['pixel_values'] = inputs['pixel_values'].to(device)
                outputs = model.get_image_features(**inputs)
                outputs /= outputs.norm(dim=-1, keepdim=True)
                img_embs.append(outputs)

        prob_matrix = []
        for i in range(len(img_embs)):
            cosine_sim = []
            for j in range(len(text_embs)):
                cosine_sim.append(img_embs[i] @ text_embs[j].T)
            prob = F.softmax(torch.tensor(cosine_sim) / temperature, dim=0)
            prob_matrix.append(prob)

        prob_matrix = torch.stack(prob_matrix)

        # marginal distribution for text embeddings
        text_emb_marginal_distribution = prob_matrix.sum(axis=0) / prob_matrix.shape[0]

        # KL divergence for each image
        image_kl_divergences = []
        for i in range(prob_matrix.shape[0]):
            kl_divergence = (prob_matrix[i, :] * torch.log(prob_matrix[i, :] / text_emb_marginal_distribution)).sum().item()
            image_kl_divergences.append(kl_divergence)

        vleu_score = np.exp(sum(image_kl_divergences) / prob_matrix.shape[0])

    model = model.to('cpu')

    return vleu_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--prompt_json_path', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_prompt', type=int)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--sdxl', default=False, action='store_true')
    args = parser.parse_args()

    if args.sdxl:
        pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to("cuda")

    model = CLIPModel.from_pretrained(args.clip_model)
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    with open(args.prompt_json_path,'r',encoding='utf-8') as f:
        prompts = json.load(f)

    final_score = get_score(pipe, model, processor, prompts[args.start:args.start+args.n_prompt], args.batch_size, args.image_dir)
    print(f'Score: {final_score}')

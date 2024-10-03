import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip


device = 'cuda'

def calculate_vleu(image_dir, prompts, temperature=0.01):
    with torch.no_grad():
        text_embs = []
        for prompt in tqdm(prompts):
            inputs = tokenizer([prompt]).to(device)
            outputs = model.encode_text(inputs)
            outputs /= outputs.norm(dim=-1, keepdim=True)
            text_embs.append(outputs)

        img_embs = []
        img_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        for img_path in tqdm(img_paths):
            image = Image.open(img_path)
            inputs = preprocess(image).unsqueeze(0).to(device)
            outputs = model.encode_image(inputs)
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
        return vleu_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--prompt_json_path', type=str)
    parser.add_argument('--image_dir', type=str)
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    with open(args.prompt_json_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    score = calculate_vleu(args.image_dir, prompts, 0.01)
    print(score)

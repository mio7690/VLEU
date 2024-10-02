import os
import re
import random
import fnmatch
import argparse
import torch
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration
import threading


def blip_caption(model_path, image_paths, device):
    # Load the model and processor
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

    for img_path in image_paths:
        if os.path.exists(img_path + '.txt'):
            continue
        # Open and process the image
        raw_image = ImageOps.exif_transpose(Image.open(img_path).convert('RGB'))

        # Conditional image captioning
        inputs = processor(raw_image, text='this is', return_tensors="pt").to(device, torch.float16)

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        caption = re.sub(r'^this is\s*','',caption).capitalize()
        print(caption)

        # Write the caption to a .txt file
        with open(img_path + '.txt', 'w') as f:
            f.write(caption)

def random_partition(input_list, num_partitions):
    random.shuffle(input_list)
    list_length = len(input_list)
    partition_size = list_length // num_partitions
    random_partitioned_lists = [input_list[i * partition_size: (i + 1) * partition_size] for i in range(num_partitions)]
    return random_partitioned_lists

def find_all_images(root_dir):
    img_files = []
    for foldername, subfolders, filenames in os.walk(root_dir):
        for extension in ['*.jpg', '*.jpeg', '*.png']:
            for filename in fnmatch.filter(filenames, extension):
                file_path = os.path.join(foldername, filename)
                img_files.append(file_path)
    return img_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, default='Salesforce/blip-image-captioning-large')
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    image_paths = find_all_images(args.data_dir)
    threads = []
    random.shuffle(image_paths)
    threads.append(threading.Thread(target=blip_caption, args=(args.model_path, image_paths, 'cuda')))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

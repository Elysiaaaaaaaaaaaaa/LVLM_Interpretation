import os
# Set the huggingface mirror and cache path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"
# 缓解显存碎片化问题
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import json
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
import argparse
import torch
from torch import nn
import torchvision.transforms.functional as TF

import numpy as np
from utils import mkdir

from tqdm import tqdm

from Advanced_IGOS_PP.utils import *
from Advanced_IGOS_PP.methods_helper import *
from Advanced_IGOS_PP.IGOS_pp import *

def parse_args():
    parser = argparse.ArgumentParser(description='Explanation for Qwen2.5-VL')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/coco/val2017',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/Qwen2.5-VL-3B-coco-caption.json',
                        help='Datasets.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/Qwen2.5-VL-3B-coco-caption/IGOS_PP',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def main(args):
    text_prompt = "Describe the image in one factual English sentence of no more than 20 words. Do not include information that is not clearly visible."
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,              # 开启 4-bit 量化
        bnb_4bit_compute_dtype=torch.float16, # 计算数据类型设为 fp16
        bnb_4bit_quant_type="nf4",      # 量化格式
        bnb_4bit_use_double_quant=True, # 二次量化，更省显存
    )
    # Load Qwen2.5-VL
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        quantization_config=quantization_config,
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # default processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    tokenizer = processor.tokenizer

    explainer = gen_explanations_qwenvl
    
    with open(args.eval_list, "r") as f:
        contents = json.load(f)
        
    save_dir = args.save_dir
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_vis_root_path = os.path.join(save_dir, "visualization")
    mkdir(save_vis_root_path)
    
    # visualization_root_path = os.path.join(save_dir, "vis")
    # mkdir(visualization_root_path)
    
    for content in tqdm(contents):
        if os.path.exists(
            os.path.join(save_npy_root_path, content["image_path"].replace(".jpg", ".npy"))
        ):
            continue
        
        image_path = os.path.join(args.Datasets, content["image_path"])
        text_prompt = content["question"]
        
        image = Image.open(image_path).convert('RGB')
        
        heatmap, superimposed_img = explainer(model, processor, image, text_prompt, tokenizer)

        # Save npy file
        np.save(
            os.path.join(save_npy_root_path, content["image_path"].replace(".jpg", ".npy")),
            np.array(heatmap)
        )
        # 保存热力图
        cv2.imwrite(os.path.join(save_vis_root_path, content["image_path"]), superimposed_img)
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
if __name__ == "__main__":
    args = parse_args()
    
    main(args)

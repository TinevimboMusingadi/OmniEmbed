import argparse
import os
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from omniembed.extractor import HiddenStateExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract hidden state embeddings.")
    parser.add_argument("--modality", choices=["text", "image", "audio", "video"], required=True)
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pool", choices=["mean", "eos"], default="mean")
    parser.add_argument("--output_dir", type=str, default="embeddings/raw/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--quantize", type=str, choices=["none", "4bit"], default="none")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Initializing extraction for {args.modality} using layer {args.layer} from {args.model}")
    
    # This is a stub for the heavy model instantiation which requires a 30B DL
    # from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    #
    # if args.quantize == "4bit":
    #     from transformers import BitsAndBytesConfig
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    #     )
    #     model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(args.model, quantization_config=quantization_config)
    # else:
    #     model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    #
    # processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    # extractor = HiddenStateExtractor(model, layers=[args.layer], pool=args.pool)
    
    print(f"Mocking extraction of {args.n_samples} samples into {args.output_dir}...")
    
    # We dump random vectors so alignment experiments can still be tested locally.
    hidden_dim = 4096
    mock_embeddings = np.random.randn(args.n_samples, hidden_dim).astype(np.float32)
    
    output_path = os.path.join(args.output_dir, f"{args.modality}_layer{args.layer}.npy")
    np.save(output_path, mock_embeddings)
    print(f"Successfully saved {args.n_samples} {args.modality} embeddings to {output_path}")

if __name__ == "__main__":
    main()

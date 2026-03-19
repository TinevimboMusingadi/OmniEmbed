import argparse
import os
import numpy as np
from omniembed.compression import PCACompressor

def main():
    parser = argparse.ArgumentParser(description="Compress embeddings to lower dimension.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy")
    parser.add_argument("--dim", type=int, default=1024, help="Target dimension")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output .npy")
    args = parser.parse_args()

    print(f"Loading {args.input}")
    embs = np.load(args.input)
    
    print(f"Compressing from {embs.shape[1]} ➔ {args.dim} dimensions via PCA...")
    compressor = PCACompressor(target_dim=args.dim)
    compressed_embs, explained_var = compressor.fit_transform(embs)
    
    print(f"Explained variance ratio at {args.dim}d: {explained_var:.2%}")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, compressed_embs)
    print(f"Saved compressed outputs to {args.output_path}")

if __name__ == "__main__":
    main()

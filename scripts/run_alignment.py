import argparse
import os
import numpy as np
from omniembed.alignment import ModalityAligner

def main():
    parser = argparse.ArgumentParser(description="Align embeddings to close modality gaps")
    parser.add_argument("--input_dir", type=str, default="embeddings/raw/")
    parser.add_argument("--output_dir", type=str, default="embeddings/aligned/")
    parser.add_argument("--method", choices=["center", "zscore", "zca"], required=True)
    args = parser.parse_args()

    print(f"Loading raw embeddings from {args.input_dir}")
    if not os.path.exists(args.input_dir):
        print("Input directory doesn't exist.")
        return

    embeddings_dict = {}
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".npy"):
            modality = filename.split("_")[0]
            embs = np.load(os.path.join(args.input_dir, filename))
            embeddings_dict[modality] = embs

    if not embeddings_dict:
        print("No .npy files found.")
        return

    aligner = ModalityAligner()
    
    if args.method == "center":
        aligned_dict = aligner.center_embeddings(embeddings_dict)
        save_folder = os.path.join(args.output_dir, "centered")
    elif args.method == "zscore":
        aligned_dict = aligner.zscore_embeddings(embeddings_dict)
        save_folder = os.path.join(args.output_dir, "zscore")
    elif args.method == "zca":
        aligner.fit_zca(embeddings_dict)
        aligned_dict = aligner.transform_zca(embeddings_dict)
        save_folder = os.path.join(args.output_dir, "whitened")
        
    os.makedirs(save_folder, exist_ok=True)
    for modality, embs in aligned_dict.items():
        out_path = os.path.join(save_folder, f"{modality}_aligned.npy")
        np.save(out_path, embs)
        print(f"Saved {args.method} aligned {modality} to {out_path}")

if __name__ == "__main__":
    main()

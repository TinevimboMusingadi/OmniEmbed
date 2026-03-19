import argparse
import os
import numpy as np
import json
from omniembed.retrieval import calculate_recall_at_k, compute_modality_gap

def main():
    parser = argparse.ArgumentParser(description="Evaluate standard retrieval/gap metrics")
    parser.add_argument("--query", type=str, required=True, help="Path to query embeddings .npy")
    parser.add_argument("--target", type=str, required=True, help="Path to target embeddings .npy")
    parser.add_argument("--output_json", type=str, default="results/metrics.json")
    args = parser.parse_args()

    print(f"Loading query: {args.query}")
    print(f"Loading target: {args.target}")

    q_embs = np.load(args.query)
    t_embs = np.load(args.target)

    if q_embs.shape[0] != t_embs.shape[0]:
        print(f"Shape mismatch: {q_embs.shape} vs {t_embs.shape}")
        return

    print("Computing metrics...")
    recalls = calculate_recall_at_k(q_embs, t_embs, k_list=[1, 5, 10])
    gaps = compute_modality_gap(q_embs, t_embs)
    
    metrics = {**recalls, **gaps}
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    
    with open(args.output_json, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(json.dumps(metrics, indent=4))
    print(f"Saved results to {args.output_json}")

if __name__ == "__main__":
    main()

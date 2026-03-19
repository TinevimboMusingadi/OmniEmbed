import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Download datasets for OmniEmbed")
    parser.add_argument("--datasets", nargs="+", choices=["coco", "audiocaps", "librispeech"], required=True)
    parser.add_argument("--output_dir", type=str, default="./data/raw")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Preparing to fetch: {args.datasets}")
    
    print("Note: In line with the updated strategy, we stream datasets "
          "directly via the HuggingFace datasets library during extraction, "
          "to avoid needing massive local storage.")

    for ds in args.datasets:
        if ds == "coco":
            print(f"COCO mapped to `nlphuji/flickr30k`")
        elif ds == "audiocaps":
            print(f"AudioCaps mapped to `d0rj/audiocaps`")
        elif ds == "librispeech":
            print(f"LibriSpeech mapped to `openslr.org`")

if __name__ == "__main__":
    main()

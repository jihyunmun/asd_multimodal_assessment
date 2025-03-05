import argparse
import os

from config import Config
from asr_decode import run_asr_decode
from extract_embedding import run_extract_embeddings
from train_loop import run_train, run_inference

#########################################################################################################
# Main
#########################################################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='Options: "decode", "embed", "train", "inference", "all"')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    mode = args.mode.lower()
    if mode == 'decode':
        run_asr_decode()
    elif mode == 'embed':
        run_extract_embeddings()
    elif mode == 'train':
        run_train()
    elif mode == 'inference':
        run_inference()
    elif mode == 'all':
        run_asr_decode()
        run_extract_embeddings()
        run_train()
        run_inference()
    else:
        print("[ERROR] Invalid mode. Choose between 'decode', 'embed', 'train', 'inference', 'all'.")

if __name__ == "__main__":
    main()
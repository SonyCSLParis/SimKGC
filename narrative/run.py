# -*- coding: utf-8 -*-
""" Run SimKGC experiments """
import os
import subprocess
import multiprocessing
import argparse
from loguru import logger

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# DS VERSIONS
VP = os.path.expanduser("~/data/SimKGC/NarrativeInductiveDataset")
VERSIONS = sorted(os.listdir(VP))

LEARNING_RATES = [1e-5, 3e-5, 5e-5]
EPOCHS = [1, 10, 50]
BATCH_SIZE = [256, 512, 1024]

def parse_args():
    parser = argparse.ArgumentParser(description="Run NarrativeInductive experiments for SimKGC")
    # Parameters for grid search
    parser.add_argument("--versions", type=str, default=None, help="Comma-separated list of versions to run")
    parser.add_argument("--learning-rates", type=str, default=None, 
                        help="Comma-separated list of learning rates to try (e.g., 0.001,0.0001)")
    parser.add_argument("--epochs", type=str, default=None,
                        help="Comma-separated list of embedding dimensions to try (e.g., 160,192)")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated list of batch sizes to try (e.g., 192,256)")
    
    return parser.parse_args()

OUTPUT_F = "./narrative/experiments"
if not os.path.exists(OUTPUT_F):
    os.makedirs(OUTPUT_F)

def main():
    args = parse_args()
    if args.versions:
        versions_to_run = args.versions.split(",")
        # Validate versions
        for v in versions_to_run:
            if v not in VERSIONS:
                logger.error(f"Version {v} not found in {VP}")
                return
    else:
        versions_to_run = VERSIONS
    if args.learning_rates:
        learning_rates = [float(lr) for lr in args.learning_rates.split(",")]
    else:
        learning_rates = LEARNING_RATES
    if args.epochs:
        epochs = [int(e) for e in args.epochs.split(",")]
    else:
        epochs = EPOCHS
    if args.batch_sizes:
        batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    else:
        batch_sizes = BATCH_SIZE

    for epoch in epochs:
        for v in VERSIONS:
            for lr in learning_rates:
                for bs in batch_sizes:
                    name = f"{v}_lr{lr}_bs{bs}_ep{epoch}"
                    output_dir = os.path.join(OUTPUT_F, name)
                    # if not os.path.exists(os.path.join(output_dir, "model_best.mdl")):
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                        data_dir = os.path.join(VP, v)
                        train_path = os.path.join(data_dir, 'train.txt.json')
                        valid_path = os.path.join(data_dir, 'valid.txt.json')
                        workers = max(1, multiprocessing.cpu_count() - 4)

                        command = f"""
                        python3 -u main.py \
                        --model-dir "{output_dir}" \
                        --pretrained-model bert-base-uncased \
                        --pooling mean \
                        --lr {lr} \
                        --use-link-graph \
                        --train-path "{train_path}" \
                        --valid-path "{valid_path}" \
                        --task "FB15k237" \
                        --batch-size {bs} \
                        --print-freq 20 \
                        --additive-margin 0.02 \
                        --use-amp \
                        --use-self-negative \
                        --finetune-t \
                        --pre-batch 2 \
                        --epochs {epoch} \
                        --workers {workers} \
                        --max-to-keep 5 "$@"
                        """
                        subprocess.run(command, shell=True, check=False)


if __name__ == "__main__":
    main()

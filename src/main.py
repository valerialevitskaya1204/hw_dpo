import argparse
import sys
from datetime import datetime

from modules.models.download_models import download_models_from_hf
from modules.load_datasets import process_dataset
from modules.training import train_model, create_training_config
from modules.utils import clean_cache, device

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
PYTHIA_MODEL_NAME = "EleutherAI/pythia-1.4b"

def train_qwen_on_tinystories(rank: int = 8):
    try:
        tokenizer, model = download_models_from_hf(
            QWEN_MODEL_NAME,
            replace_lora=True,
            rank=rank,
        )

        tokenized_train, tokenized_val = process_dataset(
            dataset_name="TinyStories",
            tokenizer=tokenizer,
            max_length=128,
            split_sizes={"train": 1000, "validation": 100},
        )

        training_config = create_training_config("qwen")

        train_model(
            model=model,
            tokenizer=tokenizer,
            tokenized_train_dataset=tokenized_train,
            tokenized_test_dataset=tokenized_val,
            model_type="qwen",
            training_config=training_config,
        )

    except Exception as e:
        raise

def train_pythia_on_hh_rlhf(rank: int = 8):
    try:
        tokenizer, model = download_models_from_hf(
            PYTHIA_MODEL_NAME,
            replace_lora=True,
            rank=rank,
        )

        tokenized_train, tokenized_val = process_dataset(
            dataset_name="hh-rlhf",
            tokenizer=tokenizer,
            max_length=1024,
        )

        training_config = create_training_config("pythia")

        train_model(
            model=model,
            tokenizer=tokenizer,
            tokenized_train_dataset=tokenized_train,
            tokenized_test_dataset=tokenized_val,
            model_type="pythia",
            training_config=training_config,
        )

    except Exception as e:
        raise

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA для Qwen (TinyStories) и Pythia (Anthropic hh-rlhf)."
    )

    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--no-pythia", action="store_true")
    parser.add_argument("--rank", type=int, default=8)

    parser.add_argument("--inference", action="store_true",
                        help="Запустить inference вместо тренировки.")
    parser.add_argument("--model", type=str, default="qwen",
                        choices=["qwen", "pythia"],
                        help="Модель для инференса.")
    parser.add_argument("--prompt", type=str, default="Tell me a story about a hat.",
                        help="Промпт для инференса.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.inference:
        model_name = QWEN_MODEL_NAME if args.model == "qwen" else PYTHIA_MODEL_NAME
        return

    start_time = datetime.now()

    if not args.no_qwen:
        train_qwen_on_tinystories(rank=args.rank)

    if not args.no_pythia:
        train_pythia_on_hh_rlhf(rank=args.rank)

    end_time = datetime.now()
    clean_cache()

if __name__ == "__main__":
    main()
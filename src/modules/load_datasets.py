from datasets import load_dataset


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def process_dataset(dataset_name, tokenizer, max_length=None, split_sizes=None):
    if dataset_name == "TinyStories":
        if split_sizes is None:
            split_sizes = {"train": 1000, "validation": 100}
        
        train_dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{split_sizes['train']}]")
        test_dataset = load_dataset("roneneldan/TinyStories", split=f"validation[:{split_sizes['validation']}]")
        
        if max_length is None:
            max_length = 128
            
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=max_length
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
            
        tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)
        
        return tokenized_train, tokenized_test
        
    elif dataset_name == "hh-rlhf":
        train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        test_dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        
        def clean_dataset(example):
            text = example['chosen'] 
            parts = text.split("Assistant:")
            input_text = "Assistant:".join(parts[:-1])
            target_text = "Assistant:" + parts[-1]
            return {"input": input_text, "target": target_text}
        
        train_dataset = train_dataset.map(clean_dataset)
        test_dataset = test_dataset.map(clean_dataset)
        
        train_dataset = train_dataset.remove_columns(['chosen', 'rejected']) 
        test_dataset = test_dataset.remove_columns(['chosen', 'rejected'])
        
        if max_length is None:
            max_length = 1024
            
        IGNORE_INDEX = -100
        tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize(examples):
            inputs = examples["input"]
            targets = examples["target"]

            all_input_ids = []
            all_attention = []
            all_labels = []

            for inp, tgt in zip(inputs, targets):
                full = inp + tgt

                enc = tokenizer(
                    full,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )

                inp_enc = tokenizer(
                    inp,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )

                input_len = sum(inp_enc["attention_mask"])

                labels = enc["input_ids"].copy()
                labels[:input_len] = [IGNORE_INDEX] * input_len

                all_input_ids.append(enc["input_ids"])
                all_attention.append(enc["attention_mask"])
                all_labels.append(labels)

            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention,
                "labels": all_labels,
            }
        
        tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
        tokenized_test = test_dataset.map(tokenize, batched=True, remove_columns=test_dataset.column_names)
        
        return tokenized_train, tokenized_test
        
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_name}")

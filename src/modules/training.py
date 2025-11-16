from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import os
import torch


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_model(model, tokenizer, tokenized_train_dataset, tokenized_test_dataset, model_type, training_config=None):
    if training_config is None:
        training_config = {}
    
    if model_type.lower() == "qwen":
        return _train_qwen(model, tokenizer, tokenized_train_dataset, tokenized_test_dataset, training_config)
    elif model_type.lower() == "pythia":
        return _train_pythia(model, tokenizer, tokenized_train_dataset, tokenized_test_dataset, training_config)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}. Поддерживаются 'qwen' и 'pythia'")

def _train_qwen(model, tokenizer, train_dataset, test_dataset, config):
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    default_args = {
        'output_dir': './lora_qwen',
        'num_train_epochs': 3,
        'per_device_train_batch_size': 4,
        'logging_dir': './logs',
        'logging_steps': 10,
        'save_steps': 500,
        'dataloader_pin_memory': True,
        'eval_steps': 500,
        'report_to': None,
        'logging_first_step': True,
        'disable_tqdm': False,
        'remove_unused_columns': False,
        'save_safetensors': False
    }

    training_args_dict = {**default_args, **config}
    
    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    return trainer.train()

def _train_pythia(model, tokenizer, train_dataset, test_dataset, config):
    
    model.config.use_cache = False 
    torch.cuda.empty_cache()
    
    max_length = config.get('max_length', 1024)
    IGNORE_INDEX = config.get('ignore_index', -100)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True, 
        max_length=max_length,
        label_pad_token_id=IGNORE_INDEX,
    )

    default_args = {
        'output_dir': './lora_pythia',
        'num_train_epochs': 1,
        'per_device_train_batch_size': 16,
        'gradient_accumulation_steps': 1,
        'dataloader_num_workers': 4,
        'learning_rate': 5e-5,
        'warmup_ratio': 0.03,
        'max_steps': 2000,
        'bf16': True,
        'fp16': False,
        'optim': "adamw_torch",
        'max_grad_norm': 0,
        'logging_dir': './logs',
        'logging_steps': 20,
        'save_steps': 2000,
        'evaluation_strategy': "no",
        'save_safetensors': False,
        'dataloader_pin_memory': True,
        'remove_unused_columns': True,
        'disable_tqdm': False,
        'report_to': None,
        'logging_first_step': True,
    }
    
    training_args_dict = {**default_args, **config}
    
    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    return trainer.train()


def create_training_config(model_type, custom_config=None):    
    base_config = {
        "qwen": {
            "output_dir": "./lora_qwen",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "learning_rate": 1e-4,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
        },
        "pythia": {
            "output_dir": "./lora_pythia", 
            "num_train_epochs": 1,
            "per_device_train_batch_size": 16,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 1,
            "max_steps": 2000,
            "bf16": True,
            "warmup_ratio": 0.03,
            "logging_steps": 20,
            "save_steps": 2000,
        }
    }
    
    if model_type.lower() not in base_config:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    config = base_config[model_type.lower()].copy()
    
    if custom_config:
        config.update(custom_config)
    
    return config
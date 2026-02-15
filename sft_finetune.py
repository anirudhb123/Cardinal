#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Dict
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import torch


class SFTConfig:
    
    DATASET_PATH = "cardinal_dataset/stackoverflow_n3000.csv"
    DATASET_DIR = "cardinal_dataset"
    DATASET_PATTERN = "*.csv"
    
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    USE_4BIT = True
    USE_FLASH_ATTENTION = True
    
    PEFT_METHOD = "lora"
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["all-linear"]
    
    OUTPUT_DIR = "./sft_output"
    NUM_EPOCHS = 3
    PER_DEVICE_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 2048
    WARMUP_STEPS = 100
    SAVE_STEPS = 500
    LOGGING_STEPS = 50
    EVAL_STEPS = 500
    
    SUBSET_SIZE = None


def load_dataset_data(config: SFTConfig) -> pd.DataFrame:
    dataset_path = Path(config.DATASET_PATH)
    
    if dataset_path.is_dir():
        dataset_dir = Path(config.DATASET_DIR)
        matching_files = list(dataset_dir.glob(config.DATASET_PATTERN))
        if not matching_files:
            raise FileNotFoundError(f"No files found matching {config.DATASET_PATTERN} in {dataset_dir}")
        dataset_path = matching_files[0]
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    if dataset_path.suffix == '.parquet':
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    
    required_cols = ["sql_text", "plan_json"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df[df["sql_text"].notna()].copy()
    df = df[df["plan_json"].notna()].copy()
    
    if config.SUBSET_SIZE is not None:
        df = df.head(config.SUBSET_SIZE)
    
    return df


def format_instruction(sql_text: str, plan_json) -> Dict[str, str]:
    instruction = "Given a SQL query, generate an optimal execution plan."
    input_text = f"SQL Query:\n{sql_text}\n\nGenerate execution plan."
    
    plan_str = json.dumps(plan_json, indent=2) if plan_json else "Plan not available"
    output_text = f"Execution Plan:\n{plan_str}"
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "text": f"{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    }


def create_dataset(df: pd.DataFrame, config: SFTConfig, tokenizer) -> Dataset:
    examples = []
    for _, row in df.iterrows():
        plan_json = row.get("plan_json")
        
        if pd.isna(plan_json):
            continue
        elif isinstance(plan_json, str):
            plan_json_str = plan_json.strip()
            if plan_json_str == "" or plan_json_str.lower() == "nan":
                continue
            else:
                try:
                    plan_json = json.loads(plan_json_str)
                except (json.JSONDecodeError, ValueError):
                    continue
        elif isinstance(plan_json, (dict, list)):
            pass
        
        example = format_instruction(
            sql_text=row["sql_text"],
            plan_json=plan_json,
        )
        examples.append(example)
    
    dataset = Dataset.from_list(examples)
    
    def tokenize_function(examples):
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            padding=False,
        )
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    initial_size = len(tokenized_dataset)
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) <= config.MAX_SEQ_LENGTH
    )
    
    return tokenized_dataset


def setup_model_and_tokenizer(config: SFTConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    bnb_config = None
    if config.USE_4BIT and config.PEFT_METHOD == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    model_kwargs = {
        "trust_remote_code": True,
    }
    
    if config.USE_FLASH_ATTENTION:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            config.USE_FLASH_ATTENTION = False
    
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        **model_kwargs
    )
    
    if config.USE_4BIT and config.PEFT_METHOD == "qlora":
        model = prepare_model_for_kbit_training(model)
    
    if config.TARGET_MODULES == ["all-linear"]:
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                target_modules.append(name.split(".")[-1])
        target_modules = list(set(target_modules))
    else:
        target_modules = config.TARGET_MODULES
    
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="SFT Fine-tuning with PEFT")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size per device")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--subset_size", type=int, help="Subset of data to use")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset file (CSV or Parquet)")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (4-bit)")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA (16-bit)")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention 2 (requires flash-attn package)")
    parser.add_argument("--no_flash_attention", action="store_true", help="Disable Flash Attention 2")
    
    args = parser.parse_args()
    
    config = SFTConfig()
    
    if args.model:
        config.MODEL_NAME = args.model
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.PER_DEVICE_BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.subset_size:
        config.SUBSET_SIZE = args.subset_size
    if args.dataset_path:
        config.DATASET_PATH = args.dataset_path
    if args.use_qlora:
        config.PEFT_METHOD = "qlora"
        config.USE_4BIT = True
    if args.use_lora:
        config.PEFT_METHOD = "lora"
        config.USE_4BIT = False
    if args.use_flash_attention:
        config.USE_FLASH_ATTENTION = True
    if args.no_flash_attention:
        config.USE_FLASH_ATTENTION = False
    
    df = load_dataset_data(config)
    
    model, tokenizer = setup_model_and_tokenizer(config)
    
    dataset = create_dataset(df, config, tokenizer)
    
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
    
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        fp16=not config.USE_4BIT,
        bf16=config.USE_4BIT,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        warmup_steps=config.WARMUP_STEPS,
        max_steps=-1,
        report_to="tensorboard",
        save_total_limit=3,
        optim="paged_adamw_8bit" if config.USE_4BIT else "adamw_torch",
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=data_collator,
    )
    
    trainer.train()
    
    trainer.save_model()
    tokenizer.save_pretrained(config.OUTPUT_DIR)


if __name__ == "__main__":
    main()

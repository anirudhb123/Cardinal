# SFT Fine-Tuning with PEFT

This directory contains scripts for supervised fine-tuning (SFT) of language models on SQL query optimization tasks using Hugging Face's PEFT library.

## Overview

The `sft_finetune.py` script fine-tunes language models on your SQLStorm dataset containing:
- **18,251 SQL queries** from StackOverflow
- **Execution plans** (PostgreSQL EXPLAIN plans)
- **Latency data** (execution times in milliseconds)

The script uses Parameter-Efficient Fine-Tuning (PEFT) techniques like **LoRA** or **QLoRA** to efficiently fine-tune large language models.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0.0`
- `transformers>=4.35.0`
- `datasets>=2.14.0`
- `peft>=0.6.0`
- `bitsandbytes>=0.41.0` (for QLoRA/4-bit quantization)
- `flash-attn` (optional, for Flash Attention 2 - see installation notes below)

### Installing Flash Attention

Flash Attention 2 requires CUDA and specific PyTorch versions. Install separately:

```bash
# For CUDA-enabled systems
pip install flash-attn --no-build-isolation

# Or build from source if needed
pip install flash-attn --no-build-isolation --no-deps
```

**Note:** Flash Attention requires:
- CUDA-enabled GPU
- PyTorch with CUDA support
- Compatible CUDA version (11.6+ for Flash Attention 2)

If Flash Attention is not available, the script will automatically fall back to standard attention.

## Quick Start

### Basic Training (QLoRA with 4-bit quantization and Flash Attention)

```bash
python sft_finetune.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --output_dir ./sft_output \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-4 \
    --task sql_to_performance \
    --use_qlora \
    --use_flash_attention
```

### Training with LoRA (16-bit)

```bash
python sft_finetune.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --output_dir ./sft_output \
    --epochs 3 \
    --batch_size 2 \
    --lr 2e-4 \
    --task sql_to_performance \
    --use_lora
```

### Training on Subset of Data (for testing)

```bash
python sft_finetune.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --output_dir ./sft_output \
    --epochs 1 \
    --batch_size 4 \
    --subset_size 1000 \
    --task sql_to_performance \
    --use_qlora
```

## Task Types

The script supports three different task formats:

### 1. `sql_to_performance`
**Input:** SQL query  
**Output:** Performance characteristics (execution time, complexity, operators, joins, etc.)

Example:
```
Input: SELECT * FROM users WHERE age > 25;
Output: Execution Time: 45.23ms
        Complexity: medium
        Operators: 5
        Joins: 0
        Aggregations: 0
```

### 2. `sql_plan_to_latency`
**Input:** SQL query + execution plan  
**Output:** Execution latency prediction

### 3. `sql_to_plan`
**Input:** SQL query  
**Output:** Optimal execution plan

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Hugging Face model name or path | `microsoft/Phi-3-mini-4k-instruct` |
| `--output_dir` | Output directory for trained model | `./sft_output` |
| `--epochs` | Number of training epochs | `3` |
| `--batch_size` | Batch size per device | `4` |
| `--lr` | Learning rate | `2e-4` |
| `--task` | Task type (`sql_to_performance`, `sql_plan_to_latency`, `sql_to_plan`) | `sql_to_performance` |
| `--subset_size` | Use subset of data (for testing) | `None` (use all data) |
| `--dataset_path` | Path to dataset file (CSV or Parquet) | `cardinal_dataset/stackoverflow_n3000.csv` |
| `--use_qlora` | Use QLoRA (4-bit quantization) | `False` |
| `--use_lora` | Use LoRA (16-bit) | `False` |
| `--use_flash_attention` | Enable Flash Attention 2 (requires flash-attn) | `True` (default) |
| `--no_flash_attention` | Disable Flash Attention 2 | `False` |

## Configuration

You can modify the `SFTConfig` class in `sft_finetune.py` to adjust:
- **Model settings:** Model name, quantization settings
- **PEFT settings:** LoRA rank (r), alpha, dropout, target modules
- **Training settings:** Epochs, batch size, learning rate, max sequence length
- **Data settings:** Filtering by execution time, dataset paths

## Dataset Preparation

The script uses the **cardinal_dataset** format which contains:
- `query`: Query filename
- `sql_text`: Full SQL query text
- `plan_json`: PostgreSQL execution plan (JSON, may be None)
- `execution_time`: Execution time in milliseconds
- `reward`: Normalized performance score

**Default dataset:** `cardinal_dataset/stackoverflow_n3000.csv` (3,000 examples)

You can specify a different dataset file using:
```bash
--dataset_path cardinal_dataset/stackoverflow_n3000.parquet
```

### Available Datasets

The script can load from:
- CSV files: `cardinal_dataset/*.csv`
- Parquet files: `cardinal_dataset/*.parquet`

### Generating Plans

If your dataset doesn't have execution plans yet, generate them using:

```bash
# First, set up PostgreSQL and load schema
psql -d stackoverflow -f v1.0/stackoverflow/schema.sql

# Then generate plans (this will create files in cardinal_dataset/)
python convert_to_cardinal.py
```

## Supported Models

The script works with any Hugging Face CausalLM model. Popular choices:

- **Small models** (recommended for testing):
  - `microsoft/Phi-3-mini-4k-instruct`
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

- **Medium models**:
  - `mistralai/Mistral-7B-Instruct-v0.2`
  - `meta-llama/Llama-2-7b-chat-hf`

- **Large models**:
  - `mistralai/Mixtral-8x7B-Instruct-v0.1`
  - `meta-llama/Llama-2-13b-chat-hf`

## Output

After training, the model is saved to the specified output directory:
```
sft_output/
├── adapter_config.json      # PEFT adapter configuration
├── adapter_model.bin        # LoRA weights
├── tokenizer_config.json    # Tokenizer configuration
├── tokenizer.json           # Tokenizer files
└── ...
```

## Monitoring Training

Training logs are saved to TensorBoard. View them with:

```bash
tensorboard --logdir ./sft_output
```

## Using the Fine-Tuned Model

After training, you can load and use the model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Load PEFT adapter
model = PeftModel.from_pretrained(base_model, "./sft_output")

# Use model for inference
prompt = "Given the following SQL query, predict its execution performance characteristics.\n\n### Input:\nSQL Query:\nSELECT * FROM users WHERE age > 25;\n\nPredict execution time and performance characteristics.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Tips

1. **Start with a subset** for testing: Use `--subset_size 1000` to test your setup before training on the full 18k dataset
2. **Use QLoRA for memory efficiency**: QLoRA allows training large models on consumer GPUs
3. **Monitor GPU memory**: Adjust batch size and gradient accumulation to fit your GPU memory
4. **Task selection**: `sql_to_performance` works without plans, while other tasks require execution plans

## Troubleshooting

**Out of Memory errors:**
- Reduce batch size (`--batch_size 2` or `1`)
- Increase gradient accumulation steps
- Use QLoRA instead of LoRA

**No plans found:**
- The script works with or without plans depending on the task type
- For `sql_to_performance`: Plans are optional
- For `sql_plan_to_latency` or `sql_to_plan`: Plans are required
- To generate plans, run `python convert_to_cardinal.py`

**Model loading errors:**
- Some models require authentication (e.g., Llama models)
- Set `HF_TOKEN` environment variable or use `huggingface-cli login`

**Flash Attention errors:**
- If `flash-attn` installation fails, use `--no_flash_attention` to disable it
- Flash Attention requires CUDA - will automatically fall back if not available
- Some models may not support Flash Attention 2 (script will warn and continue)


#!/usr/bin/env python3
"""
Batch-generate EXPLAIN (FORMAT JSON)-style plans with a local or Hugging Face model.
Writes a CSV with sql_text and the model completion (plan text).

Run from repo root:
  python scripts/generate_plans_to_csv.py --subset_size 50 --output_csv generated_plans_50.csv

Requires CUDA for reasonable speed on a 3B model (CPU works but is slow).
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Same prompt as rl_finetune_grpo.PROMPT_TEMPLATE_PLAN (plan completion mode).
PROMPT_TEMPLATE_PLAN = (
    "Generate the PostgreSQL execution plan in JSON format for the SQL query.\n"
    "Output exactly one JSON value: EXPLAIN (FORMAT JSON) shape, e.g. "
    '[{{"Plan": {{...}}}}]. Use double quotes only. No markdown fences.\n'
    "Stop immediately after the closing `]` of that JSON array. Do not add statistics, "
    "second plans, bullet lists, HTML, or any text after the JSON.\n\n"
    "[QUERY]\n{sql}\n\n[PLAN]\n"
)

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import GenerationConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=str,
        default="abharadwaj123/sqlstorm-grpo-plan8192",
        help="HF repo id or local path to merged model dir",
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        default="cardinal_dataset/stackoverflow_n3000.csv",
        help="CSV with sql_text column",
    )
    p.add_argument("--subset_size", type=int, default=50, help="First N rows after filtering empty SQL")
    p.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--repetition_penalty", type=float, default=1.08)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_4bit", action="store_true", help="Load in 4-bit (low VRAM)")
    args = p.parse_args()

    csv_path = Path(args.dataset_path)
    if not csv_path.is_file():
        raise SystemExit(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "sql_text" not in df.columns:
        raise SystemExit("CSV must have sql_text column")
    df = df[df["sql_text"].notna()].copy()
    df["sql_text"] = df["sql_text"].astype(str).str.strip()
    df = df[df["sql_text"].str.len() > 0].reset_index(drop=True)
    df = df.head(args.subset_size)

    print(f"[generate_plans] Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs: dict = dict(trust_remote_code=True, device_map="auto")
    if args.use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = "auto"

    print(f"[generate_plans] Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows: list[dict] = []
    for i in tqdm(range(len(df)), desc="generating"):
        sql = df.iloc[i]["sql_text"]
        prompt = PROMPT_TEMPLATE_PLAN.format(sql=sql)
        inputs = tokenizer(prompt, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, generation_config=gen_config)
        gen_ids = out[0, input_len:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

        row: dict = {"row_index": i}
        if "query" in df.columns:
            row["query"] = df.iloc[i]["query"]
        row["sql_text"] = sql
        row["generated_plan"] = completion
        row["model"] = args.model
        row["max_new_tokens"] = args.max_new_tokens
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, quoting=1)
    print(f"[generate_plans] Wrote {len(out_df)} rows to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()

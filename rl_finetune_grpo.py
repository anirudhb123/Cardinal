#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) fine-tuning for query optimization.

Uses the query executor to run each generated hint string against PostgreSQL
and computes a reward from three signals: inverse latency, inverse CPU time,
and inverse peak memory. The model learns to generate pg_hint_plan hints that
improve execution metrics.
"""

import os
import re
import sys
import argparse
import threading
import time
from pathlib import Path

# Ensure repo root is on path for query_execution
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load env before importing query_execution (for POSTGRES_*)
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOTrainer, GRPOConfig


class GRPOLoggingCallback(TrainerCallback):
    """Print at step start/end so we see progress during long generation phases."""

    def on_step_begin(self, args, state, control, **kwargs):
        print(f"[GRPO] Step {state.global_step}: generating completions (this can take 1–5+ min on CPU/MPS)...", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        print(f"[GRPO] Step {state.global_step}: generation done, computing rewards and updating model...", flush=True)


class Heartbeat:
    """Periodic stdout heartbeat so long generations don't look hung."""

    def __init__(self, interval_s: float = 30.0):
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def start(self, get_status):
        if self._t is not None:
            return

        def _run():
            t0 = time.time()
            while not self._stop.wait(self.interval_s):
                try:
                    status = get_status() or ""
                except Exception:
                    status = ""
                dt = int(time.time() - t0)
                print(f"[GRPO] Heartbeat: still running ({dt}s) {status}".rstrip(), flush=True)

        self._t = threading.Thread(target=_run, name="grpo-heartbeat", daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()

# Import after path and env are set
import json
from query_execution.single_query.run_query_with_metrics import run_query_with_metrics
from query_execution.single_query.plan_to_hints import plan_to_hints


# ---------- Reward: three signals ----------

def _parse_hints_from_completion(completion: str) -> str:
    """Extract pg_hint_plan hint string from model completion (e.g. /*+ ... */)."""
    if not completion or not isinstance(completion, str):
        return ""
    completion = completion.strip()
    m = re.search(r"/\*\+\s*.*?\s*\*/", completion, re.DOTALL)
    if m:
        return m.group(0).strip()
    if completion.startswith("/*+") and "*/" in completion:
        return completion.split("*/")[0] + "*/"
    return ""


def _parse_plan_from_completion(completion: str):
    """
    Extract execution plan JSON from completion (text after [PLAN] or raw JSON).
    Returns parsed plan (dict/list) or None. Compatible with EXPLAIN (FORMAT JSON) shape.
    """
    if not completion or not isinstance(completion, str):
        return None
    text = completion.strip()
    if "[PLAN]" in text.upper():
        idx = text.upper().rfind("[PLAN]")
        text = text[idx + 6:].strip()
    if not text:
        return None
    try:
        plan = json.loads(text)
    except json.JSONDecodeError:
        i = text.find("{")
        if i == -1:
            i = text.find("[")
        if i >= 0:
            depth = 0
            for j in range(i, len(text)):
                if text[j] in "{[":
                    depth += 1
                elif text[j] in "}]":
                    depth -= 1
                    if depth == 0:
                        try:
                            plan = json.loads(text[i : j + 1])
                            break
                        except json.JSONDecodeError:
                            return None
                        break
            else:
                return None
        else:
            return None
    if isinstance(plan, list) and len(plan) > 0 and isinstance(plan[0], dict):
        return plan
    if isinstance(plan, dict) and ("Plan" in plan or "Node Type" in plan):
        return plan if "Plan" in plan else {"Plan": plan}
    return None


def _plan_json_to_hints(plan_json) -> str:
    """Convert execution plan JSON to pg_hint_plan hint string."""
    if plan_json is None:
        return ""
    try:
        return plan_to_hints(plan_json) or ""
    except Exception:
        return ""


def _three_signal_reward(
    latency_ms: float,
    cpu_time_s: float,
    max_rss_kb: float,
    *,
    weight_latency: float = 1.0,
    weight_cpu: float = 1.0,
    weight_memory: float = 1.0,
    scale_latency_ms: float = 100.0,
    scale_cpu_s: float = 0.5,
    scale_memory_kb: float = 50_000.0,
) -> float:
    """
    Combine three signals into a single reward (higher is better).

    - Inverse latency: 1 / (1 + latency_ms / scale_latency_ms)
    - Inverse CPU time: 1 / (1 + cpu_time_s / scale_cpu_s)
    - Inverse memory: 1 / (1 + max_rss_kb / scale_memory_kb)

    Weights are normalized so that (weight_latency + weight_cpu + weight_memory)
    can be used to scale the total; default 1/3 each gives a combined reward in [0, 1].
    """
    total_w = weight_latency + weight_cpu + weight_memory
    if total_w <= 0:
        total_w = 1.0
    r_lat = 1.0 / (1.0 + (latency_ms or 0) / scale_latency_ms)
    r_cpu = 1.0 / (1.0 + (cpu_time_s or 0) / scale_cpu_s)
    r_mem = 1.0 / (1.0 + (max_rss_kb or 0) / scale_memory_kb)
    return (weight_latency * r_lat + weight_cpu * r_cpu + weight_memory * r_mem) / total_w


def make_reward_fn(
    db_config: dict,
    weight_latency: float = 1.0,
    weight_cpu: float = 1.0,
    weight_memory: float = 1.0,
    scale_latency_ms: float = 100.0,
    scale_cpu_s: float = 0.5,
    scale_memory_kb: float = 50_000.0,
    reward_on_error: float = 0.0,
    completion_format: str = "hints",
):
    """
    Build a GRPO reward function.
    completion_format: "hints" = model outputs /*+ ... */; "plan" = model outputs execution plan JSON, we convert to hints.
    """

    def reward_func(prompts, completions, sql_text, **kwargs):
        n_prompts = len(prompts)
        n_completions = len(completions)
        if n_prompts == 0 or n_completions == 0:
            return [reward_on_error] * max(1, n_completions)
        num_generations = n_completions // n_prompts

        print(f"[GRPO Reward] Processing batch: {n_prompts} prompts, {n_completions} completions ({num_generations} per prompt)", flush=True)

        rewards = []
        for i, completion in enumerate(completions):
            prompt_idx = i // num_generations
            sql = sql_text[prompt_idx] if isinstance(sql_text, (list, tuple)) else sql_text
            if pd.isna(sql) or not str(sql).strip():
                print(f"[GRPO Reward] idx={i}: Skipping empty SQL", flush=True)
                rewards.append(reward_on_error)
                continue
            sql = str(sql).strip()
            if completion_format == "plan":
                plan_json = _parse_plan_from_completion(completion)
                hints = _plan_json_to_hints(plan_json)
                if not hints:
                    print(f"[GRPO Reward] idx={i}: Failed to parse plan from completion (first 200 chars: {completion[:200]})", flush=True)
            else:
                hints = _parse_hints_from_completion(completion)
                if not hints:
                    print(f"[GRPO Reward] idx={i}: No hints found in completion (first 200 chars: {completion[:200]})", flush=True)

            result = run_query_with_metrics(sql, hints=hints or None, db_config=db_config, timeout_s=30.0)
            if not result.get("success"):
                error_msg = result.get("error", "unknown error")
                print(f"[GRPO Reward] idx={i}: Query failed - {error_msg}", flush=True)
                rewards.append(reward_on_error)
                continue

            latency_ms = result.get("latency_ms") or 0
            cpu_time_s = result.get("cpu_time_s") or 0
            max_rss_kb = result.get("max_rss_kb") or 0

            r = _three_signal_reward(
                latency_ms,
                cpu_time_s,
                max_rss_kb,
                weight_latency=weight_latency,
                weight_cpu=weight_cpu,
                weight_memory=weight_memory,
                scale_latency_ms=scale_latency_ms,
                scale_cpu_s=scale_cpu_s,
                scale_memory_kb=scale_memory_kb,
            )

            print(
                f"[GRPO Reward] idx={i} (prompt={prompt_idx}, gen={i % num_generations}): "
                f"latency={latency_ms:.2f}ms, cpu={cpu_time_s:.4f}s, "
                f"max_rss={max_rss_kb:.0f}KB, reward={r:.4f}",
                flush=True
            )
            rewards.append(r)

        print(f"[GRPO Reward] Batch complete: {len(rewards)} rewards computed (mean={sum(rewards)/len(rewards):.4f})", flush=True)
        return rewards

    return reward_func


# ---------- Dataset ----------

# Prompt template matching abharadwaj123/llama3-sql2plan (plan JSON output)
PROMPT_TEMPLATE_PLAN = (
    "Generate the PostgreSQL execution plan in JSON format for the SQL query.\n\n"
    "[QUERY]\n{sql}\n\n[PLAN]\n"
)
PROMPT_TEMPLATE_HINTS = (
    "Given the following SQL query, generate optimal pg_hint_plan hints to minimize execution time, CPU usage, and memory. "
    "Output only the hint comment, e.g. /*+ HashJoin(a b) SeqScan(t) */\n\nSQL:\n{sql}\n\nHints:"
)


def load_and_prepare_dataset(
    dataset_path: str,
    subset_size: int | None = None,
    prompt_template: str | None = None,
    completion_format: str = "hints",
) -> Dataset:
    """Load CSV with sql_text and build dataset with 'prompt' and 'sql_text' for GRPO."""
    if prompt_template is None:
        prompt_template = PROMPT_TEMPLATE_PLAN if completion_format == "plan" else PROMPT_TEMPLATE_HINTS
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "sql_text" not in df.columns:
        raise ValueError("Dataset must have a 'sql_text' column")
    df = df[df["sql_text"].notna()].copy()
    df["sql_text"] = df["sql_text"].astype(str).str.strip()
    df = df[df["sql_text"].str.len() > 0]
    if subset_size is not None:
        df = df.head(subset_size)
    df["prompt"] = df["sql_text"].apply(lambda s: prompt_template.format(sql=s))
    return Dataset.from_pandas(df[["prompt", "sql_text"]], preserve_index=False)


# ---------- Config ----------

class GRPOFineTuneConfig:
    DATASET_PATH = "cardinal_dataset/stackoverflow_n3000.csv"
    SUBSET_SIZE = None  # e.g. 500 for quick runs
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    SFT_CHECKPOINT = "abharadwaj123/llama3-sql2plan"  # SFT LoRA on Llama-3.2-3B
    BASE_MODEL = "meta-llama/Llama-3.2-3B"  # Required when SFT checkpoint is PEFT adapters only
    COMPLETION_FORMAT = "plan"  # "plan" = output execution plan JSON (then convert to hints); "hints" = output /*+ ... */
    USE_4BIT = False  # Set True for M1 16GB / low VRAM (loads base in 4-bit)

    OUTPUT_DIR = "./rl_grpo_output"
    NUM_EPOCHS = 1
    MAX_STEPS = -1  # -1 = use epochs; set positive when dataloader has no length (e.g. tiny subset)
    PER_DEVICE_BATCH_SIZE = 2
    NUM_GENERATIONS = 4
    MAX_PROMPT_LENGTH = 2048
    MAX_COMPLETION_LENGTH = 1024  # Plan JSON can be long; use 256 for hints-only
    LEARNING_RATE = 1e-5
    GRADIENT_ACCUMULATION_STEPS = 4

    # Reward weights (three signals)
    WEIGHT_LATENCY = 1.0
    WEIGHT_CPU = 1.0
    WEIGHT_MEMORY = 1.0
    SCALE_LATENCY_MS = 100.0
    SCALE_CPU_S = 0.5
    SCALE_MEMORY_KB = 50_000.0
    REWARD_ON_ERROR = 0.0


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with query-execution reward (latency, CPU, memory)")
    parser.add_argument("--model", type=str, help="Base model name or path")
    parser.add_argument("--sft_checkpoint", type=str, help="Path to SFT checkpoint to continue from")
    parser.add_argument("--dataset_path", type=str, help="CSV with sql_text column")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--subset_size", type=int, help="Use only first N rows")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--num_generations", type=int, help="Completions per prompt (group size)")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_latency", type=float, help="Reward weight for inverse latency")
    parser.add_argument("--weight_cpu", type=float, help="Reward weight for inverse CPU time")
    parser.add_argument("--weight_memory", type=float, help="Reward weight for inverse memory")
    parser.add_argument("--base_model", type=str, help="Base model for PEFT checkpoint (e.g. meta-llama/Llama-3.2-3B)")
    parser.add_argument("--completion_format", choices=("plan", "hints"), help="plan = output plan JSON; hints = output /*+ ... */")
    parser.add_argument("--max_completion_length", type=int, help="Max tokens per completion (default: 128 for hints, 1024 for plan)")
    parser.add_argument("--max_steps", type=int, help="Max training steps (required for tiny datasets; default -1 = use epochs)")
    parser.add_argument("--use_4bit", action="store_true", help="Load base in 4-bit for M1/low VRAM (16GB)")
    args = parser.parse_args()

    config = GRPOFineTuneConfig()
    if args.model:
        config.MODEL_NAME = args.model
    if args.sft_checkpoint:
        config.SFT_CHECKPOINT = args.sft_checkpoint
    # Only use base_model if explicitly provided (merged checkpoints don't need it)
    use_base_model = args.base_model is not None
    if use_base_model:
        config.BASE_MODEL = args.base_model
    if args.completion_format:
        config.COMPLETION_FORMAT = args.completion_format
    if args.use_4bit:
        config.USE_4BIT = True
    if args.dataset_path:
        config.DATASET_PATH = args.dataset_path
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.subset_size is not None:
        config.SUBSET_SIZE = args.subset_size
    if args.batch_size is not None:
        config.PER_DEVICE_BATCH_SIZE = args.batch_size
    if args.num_generations is not None:
        config.NUM_GENERATIONS = max(2, args.num_generations)  # GRPO requires >= 2
        if args.num_generations < 2:
            print("[GRPO] num_generations raised to 2 (GRPO minimum).", flush=True)
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.weight_latency is not None:
        config.WEIGHT_LATENCY = args.weight_latency
    if args.weight_cpu is not None:
        config.WEIGHT_CPU = args.weight_cpu
    if args.weight_memory is not None:
        config.WEIGHT_MEMORY = args.weight_memory
    if args.max_completion_length is not None:
        config.MAX_COMPLETION_LENGTH = args.max_completion_length
    elif config.COMPLETION_FORMAT == "hints":
        config.MAX_COMPLETION_LENGTH = 128  # Hints are short; keep generation fast
    if args.max_steps is not None:
        config.MAX_STEPS = args.max_steps

    model_name = config.SFT_CHECKPOINT or config.MODEL_NAME
    print(f"[GRPO] Loading tokenizer from {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("[GRPO] Tokenizer loaded.", flush=True)

    load_kwargs = dict(trust_remote_code=True, device_map="auto")
    if config.USE_4BIT:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = "auto"

    if use_base_model and config.SFT_CHECKPOINT:
        print(f"[GRPO] Loading base model {config.BASE_MODEL}...", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, **load_kwargs)
        if config.USE_4BIT:
            from peft import prepare_model_for_kbit_training
            base_model = prepare_model_for_kbit_training(base_model)
        print(f"[GRPO] Loading PEFT adapter from {config.SFT_CHECKPOINT}...", flush=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, config.SFT_CHECKPOINT)
    else:
        print(f"[GRPO] Loading merged model from {model_name} (no base model needed)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    print("[GRPO] Model loaded.", flush=True)

    print(f"[GRPO] Loading dataset from {config.DATASET_PATH}...", flush=True)
    dataset = load_and_prepare_dataset(
        config.DATASET_PATH,
        subset_size=config.SUBSET_SIZE,
        completion_format=config.COMPLETION_FORMAT,
    )
    print(f"[GRPO] Dataset loaded: {len(dataset)} examples", flush=True)

    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "database": os.getenv("POSTGRES_DB", "cardinal_test"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "your_password"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
    }
    reward_fn = make_reward_fn(
        db_config,
        weight_latency=config.WEIGHT_LATENCY,
        weight_cpu=config.WEIGHT_CPU,
        weight_memory=config.WEIGHT_MEMORY,
        scale_latency_ms=config.SCALE_LATENCY_MS,
        scale_cpu_s=config.SCALE_CPU_S,
        scale_memory_kb=config.SCALE_MEMORY_KB,
        reward_on_error=config.REWARD_ON_ERROR,
        completion_format=config.COMPLETION_FORMAT,
    )

    # GRPO requires >= 2 generations; enforce before creating GRPOConfig
    num_gen = max(2, config.NUM_GENERATIONS)
    if config.NUM_GENERATIONS < 2:
        print("[GRPO] num_generations set to 2 (GRPO minimum).", flush=True)
        config.NUM_GENERATIONS = num_gen

    # GRPO's RepeatSampler uses batch_size = (per_device * world_size * steps_per_generation) // num_generations.
    # With 1 example and default gradient_accumulation_steps=4, steps_per_generation=4 → sampler batch_size=2 → 1//2=0 batches.
    # Use gradient_accumulation_steps=2 so steps_per_generation=2 → sampler batch_size=1 → at least one batch.
    if len(dataset) <= 2 and config.GRADIENT_ACCUMULATION_STEPS > 2:
        config.GRADIENT_ACCUMULATION_STEPS = 2
        print("[GRPO] gradient_accumulation_steps set to 2 so GRPO dataloader yields batches with tiny dataset.", flush=True)

    # When dataset is tiny, dataloader may report no length; set max_steps so training runs.
    max_steps = config.MAX_STEPS
    if max_steps <= 0 and len(dataset) > 0:
        steps_per_epoch = max(1, len(dataset) // (config.PER_DEVICE_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS))
        max_steps = steps_per_epoch * config.NUM_EPOCHS
        print(f"[GRPO] Inferred max_steps={max_steps} from dataset size and epochs.", flush=True)
    if max_steps <= 0:
        max_steps = 1
        print("[GRPO] Using max_steps=1 (minimal run).", flush=True)

    # On CPU/MPS (e.g. M1), bf16 is not supported; use fp32 and optionally use_cpu
    use_cpu = not torch.cuda.is_available()
    use_bf16 = torch.cuda.is_available()
    if use_cpu:
        print("[GRPO] No CUDA detected; using CPU (fp32).", flush=True)
        # Avoid multiprocessing in reward (subprocess can't pickle local targets in some envs)
        os.environ["SQLSTORM_QUERY_IN_PROCESS"] = "1"

    print(f"[GRPO] Config: num_generations={num_gen}, max_completion_length={config.MAX_COMPLETION_LENGTH}, batch_size={config.PER_DEVICE_BATCH_SIZE}, max_steps={max_steps}", flush=True)

    training_args = GRPOConfig(
        output_dir=config.OUTPUT_DIR,
        max_steps=max_steps,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        num_generations=num_gen,
        max_completion_length=config.MAX_COMPLETION_LENGTH,
        remove_unused_columns=False,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=use_bf16,
        use_cpu=use_cpu,
        gradient_checkpointing=True,
    )

    print(f"[GRPO] Initializing trainer with {len(dataset)} examples...", flush=True)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        callbacks=[GRPOLoggingCallback()],
    )
    print("[GRPO] Starting training... (first step = generation only; expect 1–5+ min before reward logs)", flush=True)
    hb = Heartbeat(interval_s=30.0)
    hb.start(lambda: f"(global_step={getattr(trainer.state, 'global_step', '?')})")
    trainer.train()
    hb.stop()
    print(f"[GRPO] Training complete. Saving to {config.OUTPUT_DIR}", flush=True)
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print("[GRPO] Done!", flush=True)


if __name__ == "__main__":
    main()

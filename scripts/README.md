# Slurm scripts

## SFT fine-tuning

Run SFT on a Slurm cluster:

```bash
# From repo root
mkdir -p logs
sbatch scripts/slurm_sft.sbatch
```

### Resource defaults (edit in `slurm_sft.sbatch`)

- 1 GPU, 8 CPUs, 32G RAM, 24h
- Partition: `gpu` (change `#SBATCH --partition` for your cluster)
- Outputs: `logs/slurm_sft_<jobid>.out`, `logs/slurm_sft_<jobid>.err`

### Multi-GPU

In `slurm_sft.sbatch` set e.g.:

```bash
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
```

The script uses `accelerate launch` and will use all allocated GPUs.

### Overriding training options

Use environment variables when submitting:

```bash
SFT_OUTPUT_DIR=/path/to/output \
SFT_DATASET_PATH=cardinal_dataset/stackoverflow_n3000.csv \
SFT_EXTRA_ARGS="--subset_size 500 --epochs 2 --no_flash_attention" \
  sbatch scripts/slurm_sft.sbatch
```

### One-time setup

If `accelerate` has never been configured, run once interactively (e.g. on a login node):

```bash
accelerate config
```

Choose defaults (single machine, multi-GPU if needed, no DeepSpeed unless you want it). Or the script will use inline `--num_processes` and typical defaults.

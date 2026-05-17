# Aubo i10 Inspire VLA Pipeline (ACT First)

## 1) Collect data

```bash
source .venv/bin/activate
python -m src.core.collect_vla_dataset \
  --dataset-root /path/to/dataset \
  --episodes-per-task 30
```

Tasks are cycled in this order: `reach`, `grasp`, `place`.

## 2) Check dataset quality

```bash
source .venv/bin/activate
python -m src.dataset.check_dataset_quality \
  --dataset-root /path/to/dataset
```

## 3) Launch ACT training

```bash
source .venv/bin/activate
python -m src.core.train_act \
  --dataset-repo-id local/aubo-i10-vla \
  --dataset-root /path/to/dataset \
  --output-dir outputs/train/act_aubo_i10 \
  --job-name act_aubo_i10 \
  --device cuda \
  --batch-size 16 \
  --steps 120000 \
  --chunk-size 20 \
  --n-action-steps 20
```

If you need to inspect the generated `lerobot-train` command first:

```bash
source .venv/bin/activate
python -m src.core.train_act \
  --dataset-repo-id local/aubo-i10-vla \
  --dataset-root /path/to/dataset \
  --dry-run
```

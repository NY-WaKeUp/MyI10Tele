# MyI10Tele

This project is now unified on a single `dm_control` backend environment:

- Core environment: `src/core/mujoco_teleop_env.py`
- Data collection: `src/core/collect_vla_dataset.py`
- Teleop collection: `src/core/teleop_collect_v3.py`
- Compatibility wrapper: `src/core/MyEnv.py`

## Quick start

```bash
cd /Users/ningyu/code_before_paper/MyI10Tele
uv venv --seed
source .venv/bin/activate
uv sync
python src/core/collect_vla_dataset.py --help
```

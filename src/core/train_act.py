import argparse
import shlex
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ACT training with lerobot-train.")
    parser.add_argument("--dataset-repo-id", type=str, required=True, help="Dataset repo id or local name.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Local dataset root.")
    parser.add_argument("--output-dir", type=str, default="outputs/train/act_aubo_i10_inspire")
    parser.add_argument("--job-name", type=str, default="act_aubo_i10_inspire")
    parser.add_argument("--policy-repo-id", type=str, default="local/act_aubo_i10_inspire")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--n-action-steps", type=int, default=20)
    parser.add_argument("--wandb-enable", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="act_aubo_i10_inspire")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={args.dataset_root}",
        "--policy.type=act",
        f"--policy.device={args.device}",
        f"--policy.repo_id={args.policy_repo_id}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--num_workers={args.num_workers}",
        f"--wandb.enable={'true' if args.wandb_enable else 'false'}",
        f"--wandb.project={args.wandb_project}",
        "--eval_freq=-1",
    ]
    if args.wandb_enable and args.wandb_entity:
        cmd.append(f"--wandb.entity={args.wandb_entity}")

    pretty_cmd = " ".join(shlex.quote(c) for c in cmd)
    print("[train_act] launching command:")
    print(pretty_cmd)
    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

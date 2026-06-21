"""Tyro CLI for ``8.val_openpi_sim.py`` — grouped flags + flat aliases for scripts."""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any

import tyro

from core.dataset_config import AUBOI10_QPOS_ROOT_V21_CORRECT, TASK_NAME, XML_PATH

_A = tyro.conf.arg


class ActionType(str, Enum):
    """Which action space the policy outputs (must match train config + dataset)."""

    qpos = "qpos"
    ee_pose = "ee_pose"


class PolicyObsSource(str, Enum):
    """Where policy inputs (image + state) come from at infer time."""

    sim = "sim"
    dataset = "dataset"
    # Tyro CLI uses member names (underscores); EvalArgs flattens to hyphen strings.
    state_dataset_image_sim = "state-dataset-image-sim"
    state_sim_image_dataset = "state-sim-image-dataset"


@dataclass
class ServerConfig:
    """openpi ``serve_policy`` WebSocket endpoint."""

    host: Annotated[
        str,
        _A(
            aliases=["--host"],
            help="openpi policy server host (``serve_policy --port`` 同一机器填 localhost).",
        ),
    ] = "localhost"
    port: Annotated[
        int,
        _A(
            aliases=["--port"],
            help="openpi policy server 端口，与 ``serve_policy`` 一致.",
        ),
    ] = 8000


@dataclass
class RolloutConfig:
    """MuJoCo rollout episodes and scene."""

    num_episodes: Annotated[
        int,
        _A(
            aliases=["--num-episodes"],
            help="评测跑几条 episode（每条 reset 一次）.",
        ),
    ] = 10
    max_steps: Annotated[
        int,
        _A(
            aliases=["--max-steps"],
            help="单条 episode 最多多少个 20Hz 控制步；超时记为 failure.",
        ),
    ] = 600
    seed: Annotated[
        int,
        _A(
            aliases=["--seed"],
            help="MyEnv / 场景采样 RNG seed（影响 reset 随机性）.",
        ),
    ] = 42
    prompt: Annotated[
        str,
        _A(
            aliases=["--prompt"],
            help="发给 policy 的语言指令；需与 serve 侧 ``--default-prompt`` 语义一致.",
        ),
    ] = TASK_NAME
    xml_path: Annotated[
        str,
        _A(
            aliases=["--xml-path"],
            help="MuJoCo 场景 XML 路径.",
        ),
    ] = XML_PATH
    action_type: Annotated[
        ActionType,
        _A(
            aliases=["--action-type"],
            help="qpos=关节目标；ee_pose=绝对法兰 xyz+rpy+gripper. 须与 checkpoint TrainConfig 一致.",
        ),
    ] = ActionType.qpos
    teleop_render: Annotated[
        bool,
        _A(
            aliases=["--teleop-render"],
            help="MuJoCo viewer 叠加相机小窗 + 按键提示（与 0.tele.py 同款 overlay）；"
            "仅影响显示，不改 policy/物理.",
        ),
    ] = False
    log_cube_every: Annotated[
        int,
        _A(
            aliases=["--log-cube-every"],
            help="每 N 个控制步打印一次 cube xyz；0=仅在 cube 离桌时打印.",
        ),
    ] = 25


@dataclass
class PolicyConfig:
    """Policy I/O, replanning, and qpos closed-loop options."""

    replan_steps: Annotated[
        int,
        _A(
            aliases=["--replan-steps"],
            help="仅 action_delta_stride=1 时有效：一次 infer 后连续执行 chunk[0..R-1] 再 infer."
            " R=1 每步 infer；R=30 则 30 步开环. K>1 时必须为 1.",
        ),
    ] = 1
    action_delta_stride: Annotated[
        int,
        _A(
            aliases=["--action-delta-stride"],
            help="须与训练 config 的 action_delta_stride 一致."
            " K=1：逐步目标；K=10(k10)：每 K 步 infer，只 hold chunk[0]（≈t+K 目标）.",
        ),
    ] = 1
    policy_obs_source: Annotated[
        PolicyObsSource,
        _A(
            aliases=["--policy-obs-source"],
            help="policy 输入来源. sim=全仿真；dataset=全 LeRobot；"
            "state_dataset_image_sim=Hybrid A(state 录数/sim 图)；"
            "state_sim_image_dataset=Hybrid B. CLI 用下划线枚举名.",
        ),
    ] = PolicyObsSource.sim
    policy_obs_dataset_episode: Annotated[
        int,
        _A(
            aliases=["--policy-obs-dataset-episode"],
            help="当 obs 需读 LeRobot 时，用 ``--lerobot-root`` 下第几条 episode 的帧"
            "（0-based，如 11=第 12 条 demo）作为 policy 的 image/state 流."
            "若未设 dataset_init_episode 会自动对齐同号.",
        ),
    ] = 0
    policy_obs_state_alpha: Annotated[
        float,
        _A(
            aliases=["--policy-obs-state-alpha"],
            help="仅 policy_obs_source=sim：proprio 混合 "
            "(1-A)*sim_qpos + A*demo_qpos[step]，用于 A/B 漂移实验. 0=纯 sim.",
        ),
    ] = 0.0
    qpos_hold_ramp: Annotated[
        bool,
        _A(
            aliases=["--qpos-hold-ramp"],
            help="k10 持有 chunk[0] 时：线性插值 anchor→target（按 hold 索引）；"
            "默认 False=10 步恒定 chunk[0].",
        ),
    ] = False
    qpos_chunk_integrate: Annotated[
        bool,
        _A(
            aliases=["--qpos-chunk-integrate"],
            help="与 qpos_receding_horizon 联用：cmd_arm = sim_q + (chunk[0]-infer_q)，"
            "把绝对预测转成相对当前 sim 的增量，抗 proprio drift.",
        ),
    ] = False
    qpos_receding_horizon: Annotated[
        bool,
        _A(
            aliases=["--qpos-receding-horizon"],
            help="stride=1 时每步 infer，且每步只用 chunk[0]（不用 chunk[1..]）.",
        ),
    ] = False
    qpos_exec_via_ik: Annotated[
        bool,
        _A(
            aliases=["--qpos-exec-via-ik"],
            help="实验：qpos 输出经 FK→ capped EE delta→IK 执行（仿 teleop 链）."
            "仅 stride=1；与 qpos_hold_ramp 不兼容.",
        ),
    ] = False
    ee_guard: Annotated[
        bool,
        _A(
            aliases=["--ee-guard"],
            help="ee_pose 时默认 ON：限制每步 |Δxyz|/|Δrpy|，防 policy 一步跳太大."
            "``--no-ee-guard`` 关闭.",
        ),
    ] = True
    max_ee_xyz_step: Annotated[
        float,
        _A(
            aliases=["--max-ee-xyz-step"],
            help="EE guard 每步最大平移 (m)，默认 0.005≈键盘 teleop.",
        ),
    ] = 0.005
    max_ee_rpy_step: Annotated[
        float,
        _A(
            aliases=["--max-ee-rpy-step"],
            help="EE guard 每步最大旋转 (rad).",
        ),
    ] = 0.08


@dataclass
class PhysicsConfig:
    """Per-tick physics stepping (default = 0.tele 20 Hz continuous, 0 extra mj_step)."""

    teleop_tick: Annotated[
        bool,
        _A(
            aliases=["--teleop-tick"],
            help="默认 ON：与 0.tele 对齐，step(action) 后不再额外 mj_step，"
            "仅外层 while 的 step_env 推进 20Hz. ``--no-teleop-tick`` 走 settle ablation.",
        ),
    ] = True
    physics_settle_steps: Annotated[
        int,
        _A(
            aliases=["--physics-settle-steps"],
            help="``--no-teleop-tick`` 且未设 tol 时：每步 action 后固定 mj_step 次数."
            "GT replay 常用；policy 连续模式默认不用.",
        ),
    ] = 50
    physics_settle_tol: Annotated[
        float,
        _A(
            aliases=["--physics-settle-tol"],
            help="``--no-teleop-tick`` 且 CLI 显式传入时：自适应 settle 直到 "
            "||q_arm - q_target|| <= tol (rad). 会击穿严格 20Hz.",
        ),
    ] = 0.0005
    physics_settle_max_steps: Annotated[
        int,
        _A(
            aliases=["--physics-settle-max-steps"],
            help="自适应 settle 的 mj_step 上限.",
        ),
    ] = 1000
    post_action_wait_s: Annotated[
        float,
        _A(
            aliases=["--post-action-wait-s"],
            help=">0 启用 wait-replan：step→等待 N 秒仿真时间→再观测 infer（每步一次）."
            "与 teleop_tick 互斥；用于「等稳再推理」实验.",
        ),
    ] = 0.0


@dataclass
class DatasetConfig:
    """LeRobot roots, GT replay, and dataset-aligned init."""

    lerobot_root: Annotated[
        str,
        _A(
            aliases=["--lerobot-root"],
            help="LeRobot 数据集根目录；GT replay / dataset obs / init 都读这里."
            " qpos 与 ee_pose 评测须各自对应正确的 root.",
        ),
    ] = AUBOI10_QPOS_ROOT_V21_CORRECT
    dataset_init_episode: Annotated[
        int | None,
        _A(
            aliases=["--dataset-init-episode"],
            help="每条 rollout 前从该 episode 的 frame-0 恢复 obj_init + 初始 qpos，"
            "与录数布局对齐. None=随机 reset.",
        ),
    ] = None
    dataset_init_warmup_steps: Annotated[
        int,
        _A(
            aliases=["--dataset-init-warmup-steps"],
            help="snap 到 frame-0 qpos 后再跑 N 次 step_env（让 PD 稳定）.",
        ),
    ] = 20
    replay_gt_episode: Annotated[
        int | None,
        _A(
            aliases=["--replay-gt-episode"],
            help="若设置：不开 policy，开环回放该 episode 的 parquet actions（诊断执行层）.",
        ),
    ] = None
    replay_max_frames: Annotated[
        int | None,
        _A(
            aliases=["--replay-max-frames"],
            help="GT replay 最多回放多少帧；None=整条 episode.",
        ),
    ] = None
    replay_dataset_init: Annotated[
        bool,
        _A(
            aliases=["--replay-dataset-init"],
            help="GT replay 时是否用 parquet 里的 obj_init 恢复 cube/平台."
            "``--no-replay-dataset-init`` 关闭.",
        ),
    ] = True


@dataclass
class TraceConfig:
    """Rollout traces, videos, and offline analysis."""

    trace_dir: Annotated[
        str | None,
        _A(
            aliases=["--trace-dir"],
            help="保存 npz/json trace、summary、action_chunks 的目录；"
            "设了则默认 video 写到 ``<trace-dir>/videos``.",
        ),
    ] = None
    trace_episodes: Annotated[
        int,
        _A(
            aliases=["--trace-episodes"],
            help="只 trace 前 N 条 episode；0=全部.",
        ),
    ] = 0
    trace_analyze_only: Annotated[
        bool,
        _A(
            aliases=["--trace-analyze-only"],
            help="只读已有 trace_dir 打印统计并退出（不连 policy、不开 sim rollout）.",
        ),
    ] = False
    trace_plot_only: Annotated[
        bool,
        _A(
            aliases=["--trace-plot-only"],
            help="只从 trace_dir 的 rollout.npz 重画 qpos 时间曲线并退出.",
        ),
    ] = False
    video_dir: Annotated[
        str | None,
        _A(
            aliases=["--video-dir"],
            help=" rollout 视频输出目录；默认 trace_dir/videos 或 ./episode_videos_openpi.",
        ),
    ] = None


@dataclass
class EvalCLI:
    """openpi WebSocket eval on Aubo MyEnv simulation.

    Examples
    --------
    Sim closed-loop (20 Hz, matches 0.tele.py)::

        PYTHONPATH=src python src/core/8.val_openpi_sim.py \\
          --port 8000 --action-type qpos --num-episodes 20 \\
          --dataset-init-episode 0 --trace-dir ./openpi_eval_trace

    GT open-loop replay (no policy server)::

        PYTHONPATH=src python src/core/8.val_openpi_sim.py \\
          --replay-gt-episode 0 --action-type qpos --trace-dir ./openpi_gt_replay

    Trace analysis only::

        PYTHONPATH=src python src/core/8.val_openpi_sim.py \\
          --trace-dir ./openpi_eval_trace --trace-analyze-only
    """

    server: ServerConfig = field(default_factory=ServerConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)


class EvalArgs:
    """Flat namespace consumed by ``8.val_openpi_sim.py`` (backward compatible)."""

    def __init__(self, cfg: EvalCLI) -> None:
        self._cfg = cfg
        for section in (
            cfg.server,
            cfg.rollout,
            cfg.policy,
            cfg.physics,
            cfg.dataset,
            cfg.trace,
        ):
            for f in dataclasses.fields(section):
                val = getattr(section, f.name)
                if isinstance(val, Enum):
                    val = val.value
                setattr(self, f.name, val)

    def to_json_dict(self) -> dict[str, Any]:
        return _enum_to_value(dataclasses.asdict(self._cfg))


def _enum_to_value(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Enum):
            out[k] = v.value
        elif isinstance(v, dict):
            out[k] = _enum_to_value(v)
        else:
            out[k] = v
    return out


def physics_settle_tol_on_cli(argv: list[str] | None = None) -> bool:
    argv = argv if argv is not None else sys.argv[1:]
    keys = (
        "--physics-settle-tol",
        "--physics.physics-settle-tol",
        "--physics.physics_settle_tol",
    )
    for a in argv:
        for k in keys:
            if a == k or a.startswith(f"{k}="):
                return True
    return False


def parse_eval_args(argv: list[str] | None = None) -> EvalArgs:
    cfg = tyro.cli(EvalCLI, args=argv)
    return EvalArgs(cfg)

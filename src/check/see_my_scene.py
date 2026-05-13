# Preview MJCF in passive viewer; optional RGB from onboard cameras (agentview + wrist_cam).

import argparse
import multiprocessing as mp
import os
import queue
import sys

import cv2
import mujoco
from mujoco import viewer
import numpy as np
from pathlib import Path

asset_dir = Path(__file__).parents[2] / "assets" / "aubo_i10_inspire"
from core import opencv_preview_worker

# Arm joints (rad); cannot use <keyframe> in robot MJCF when this file is merged into a larger scene (nq mismatch).
AUBO_I10_HOME_ARM_RAD = (
    -1.0568138360977173,
    -0.48808249831199646,
    1.3184903860092163,
    0.22961488366127014,
    1.5413566827774048,
    0.5091112852096558,
)

# MODEL_PATH = os.path.join(_HERE, "aubo_i10_30_inspire.xml")
# MODEL_PATH = os.path.join(_ASSET, "aubo_i10_2/aubo_i10.xml")
MODEL_PATH = asset_dir / "myscene.xml"
# MODEL_PATH = os.path.join(_ASSET, "/Users/ningyu/code_before_paper/mujoco-learning/model/franka_emika_panda/panda.xml")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Passive MuJoCo viewer + optional onboard camera preview."
    )
    p.add_argument(
        "--no-camera-windows",
        action="store_true",
        help="Disable OpenCV camera preview window.",
    )
    p.add_argument(
        "--cam-width", type=int, default=320, help="Width per camera panel (pixels)."
    )
    p.add_argument(
        "--cam-height", type=int, default=240, help="Height per camera panel (pixels)."
    )
    p.add_argument(
        "--camera-display",
        choices=("auto", "inline", "process"),
        default="auto",
        help="OpenCV GUI: on macOS, mjpython(GLFW) conflicts with cv2 in-process; 'auto' uses a subprocess there.",
    )
    return p.parse_args()


def _fixed_scene_camera(cam_id: int) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    return cam


def _apply_aubo_home_arm(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    names = [
        "shoulder_joint",
        "upperArm_joint",
        "foreArm_joint",
        "wrist1_joint",
        "wrist2_joint",
        "wrist3_joint",
    ]
    for name, q in zip(names, AUBO_I10_HOME_ARM_RAD):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint '{name}' not found in model")
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = q


def main() -> None:
    args = _parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    _apply_aubo_home_arm(model, data)
    mujoco.mj_forward(model, data)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    i10_inspire_flange_link_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "i10_inspire_flange_link"
    )
    wrist3_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "wrist3_joint"
    )
    if wrist3_joint_id < 0:
        raise ValueError("joint 'wrist3_joint' not found in model")
    if ee_site_id < 0:
        raise ValueError("site 'ee_site' not found in model")

    agentview_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    show_cams = not args.no_camera_windows and agentview_id >= 0 and wrist_cam_id >= 0
    if not args.no_camera_windows:
        if agentview_id < 0:
            print(
                "warn: camera 'agentview' not found; disable preview or add table.xml camera."
            )
        if wrist_cam_id < 0:
            print(
                "warn: camera 'wrist_cam' not found; disable preview or add wrist_cam in aubo_i10_inspire.xml."
            )

    renderer = None
    if show_cams:
        renderer = mujoco.Renderer(model, width=args.cam_width, height=args.cam_height)

    display_mode = args.camera_display
    if display_mode == "auto":
        display_mode = "process" if sys.platform == "darwin" else "inline"

    frame_queue: mp.Queue | None = None
    preview_proc: mp.Process | None = None
    if show_cams and display_mode == "process":
        frame_queue = mp.Queue(maxsize=1)
        preview_proc = mp.Process(
            target=opencv_preview_worker.run, args=(frame_queue,), daemon=False
        )
        preview_proc.start()

    print("model =", MODEL_PATH)
    print("ngeom =", model.ngeom, "nu =", model.nu)
    if show_cams:
        how = "subprocess OpenCV" if display_mode == "process" else "in-process OpenCV"
        print(
            f"camera preview ({how}): agentview | wrist_cam — close MuJoCo viewer to quit"
        )

    cam_agent = _fixed_scene_camera(agentview_id) if show_cams else None
    cam_wrist = _fixed_scene_camera(wrist_cam_id) if show_cams else None

    try:
        with viewer.launch_passive(model, data) as v:
            v.cam.azimuth = 0
            v.cam.elevation = -10
            v.cam.distance = 2.0
            v.cam.lookat[:] = np.array([0.4, -1.0, 0.43], dtype=np.float64)
            with v.lock():
                v.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            step_count = 0
            while v.is_running():
                _apply_aubo_home_arm(model, data)
                mujoco.mj_step(model, data)
                if step_count % 60 == 0:
                    ee_pos = data.site_xpos[ee_site_id]
                    print(
                        f"ee_site xyz (m): [{ee_pos[0]: .4f}, {ee_pos[1]: .4f}, {ee_pos[2]: .4f}]"
                    )
                    i10_inspire_flange_link_pos = data.xpos[i10_inspire_flange_link_id]
                    print(
                        f"i10_inspire_flange_link xyz (m): [{i10_inspire_flange_link_pos[0]: .4f}, {i10_inspire_flange_link_pos[1]: .4f}, {i10_inspire_flange_link_pos[2]: .4f}]"
                    )
                    wrist3_anchor = data.xanchor[wrist3_joint_id]
                    print(
                        f"wrist3_anchor xyz (m): [{wrist3_anchor[0]: .4f}, {wrist3_anchor[1]: .4f}, {wrist3_anchor[2]: .4f}]"
                    )
                step_count += 1

                if (
                    show_cams
                    and renderer is not None
                    and cam_agent is not None
                    and cam_wrist is not None
                ):
                    renderer.update_scene(data, camera=cam_agent)
                    rgb_agent = renderer.render()
                    renderer.update_scene(data, camera=cam_wrist)
                    rgb_wrist = renderer.render()
                    panel_a = cv2.cvtColor(rgb_agent, cv2.COLOR_RGB2BGR)
                    panel_w = cv2.cvtColor(rgb_wrist, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        panel_a,
                        "agentview",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        panel_w,
                        "wrist_cam",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    preview = np.ascontiguousarray(np.hstack([panel_a, panel_w]))
                    if frame_queue is not None:
                        try:
                            frame_queue.put_nowait(preview)
                        except queue.Full:
                            pass
                    else:
                        cv2.imshow("onboard_cameras", preview)
                        cv2.waitKey(1)

                v.sync()
    finally:
        if frame_queue is not None:
            try:
                frame_queue.put(None, timeout=0.5)
            except Exception:
                pass
            if preview_proc is not None:
                preview_proc.join(timeout=4.0)
                if preview_proc.is_alive():
                    preview_proc.terminate()
                    preview_proc.join(timeout=1.0)
        else:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

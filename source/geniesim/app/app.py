# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import geniesim.utils.system_utils as system_utils
from geniesim.config.params import *

system_utils.check_and_fix_env()

ps = ParameterServer()
for f in fields(Config):
    ps.declare_parameter(f.name, None)
ps.set_parameters_from_yaml(system_utils.config_path() + "/config.yaml")
ps.override_from_cli()
cfg = load_dataclass(Config, ps)


from geniesim.app.workflow import AppLauncher

app_launcher = AppLauncher(cfg.app)
simulation_app = app_launcher.app

import carb
carb.settings.get_settings().set(
    "/rtx/raytracing/fractionalCutoutOpacity", True
)
import omni


# Global variables
import queue
import threading
import time

_frame_count = 0
_last_time = time.time()

# ── EmbodiedClaw HTTP Frame Server ────────────────────────────────────────────
_frame_bytes: bytes = b""
_frame_lock = threading.Lock()
FRAME_SERVER_PORT = 9000
_debug_info: dict = {}
_viewport_annotator = None  # 懒初始化，首次调用时创建
_api_core = None  # 由 main() 设置，供 HTTP handler 调用
_skill_executor = None  # 由 main() 设置，供 HTTP handler 调用
_move_arm_executor: "MoveArmExecutor | None" = None  # 由 main() 设置，供 HTTP handler 调用
_left_curobo_motion = None  # 左臂 CuroboMotion，首次左臂请求时懒初始化
_left_curobo_init_lock = threading.Lock()

# 头部相机（供 AnyGrasp 使用，RGB+Depth 对齐）
_head_rgb_bytes: bytes = b""
_head_depth_bytes: bytes = b""
_head_lock = threading.Lock()
_head_depth_annotator = None   # distance_to_image_plane annotator，懒初始化
_head_camera_info: dict = {}   # {fx, fy, cx, cy, width, height}，懒初始化


class SkillExecutor:
    """事件驱动状态机，让 EmbodiedClaw 通过 HTTP 触发 VLA 技能执行并等待结果。"""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"

    def __init__(self):
        self._state = self.IDLE
        self._instruction: str = ""
        self._lock = threading.Lock()
        self._start_event = threading.Event()
        self._done_event = threading.Event()

    def request(self, instruction: str):
        """HTTP handler 调用：发起技能请求（非阻塞）。"""
        with self._lock:
            self._instruction = instruction
            self._state = self.RUNNING
        self._done_event.clear()
        self._start_event.set()

    def wait_for_start(self) -> str:
        """benchmark loop 调用：阻塞等待技能请求，返回指令。"""
        self._start_event.wait()
        self._start_event.clear()
        with self._lock:
            return self._instruction

    def complete(self, success: bool):
        """benchmark loop 调用：技能执行完毕，上报结果。"""
        with self._lock:
            self._state = self.SUCCESS if success else self.FAILURE
        self._done_event.set()

    def wait_for_done(self, timeout: float = 120.0) -> bool:
        """HTTP handler 调用：阻塞等待技能完成，返回 True 表示在超时内完成。"""
        return self._done_event.wait(timeout=timeout)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def reset(self):
        with self._lock:
            self._state = self.IDLE
        self._start_event.clear()
        self._done_event.clear()


class MoveArmExecutor:
    """Thread-safe request/response bridge: HTTP handler → cuRobo worker thread."""

    def __init__(self):
        self._req_q: queue.Queue = queue.Queue(maxsize=1)
        self._res_q: queue.Queue = queue.Queue()

    def request(self, position, quat_wxyz, arm: str = "right", timeout: float = 90.0):
        """HTTP handler: submit request, block until done or timeout.
        Returns (success: bool, error: str).
        """
        try:
            self._req_q.put_nowait({"position": list(position), "quat_wxyz": list(quat_wxyz), "arm": arm})
        except queue.Full:
            return False, "move_arm busy"
        try:
            ok, err = self._res_q.get(timeout=timeout)
            return ok, err
        except queue.Empty:
            return False, "timeout"

    def get_next(self, timeout: float = 0.2):
        """Worker thread: get next pending request. Returns None if queue empty."""
        try:
            return self._req_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def complete(self, success: bool, error: str = ""):
        """Worker thread: signal completion to HTTP handler."""
        self._res_q.put((success, error))


def _rotate_quat_by_matrix(q, R):
    """Rotate quaternion q (wxyz) by 3x3 rotation matrix R.

    Returns the quaternion representing R * q (i.e., compose R then q, or
    equivalently: express orientation q (local frame) in the frame defined by R).
    """
    import numpy as np

    # Convert R to quaternion qR (wxyz)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qR = np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qR = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qR = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qR = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    qR = qR / np.linalg.norm(qR)

    # Quaternion multiply qR * q  (both wxyz)
    w1, x1, y1, z1 = qR
    w2, x2, y2, z2 = q
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _move_arm_worker():
    """Background thread: process /move_arm requests via cuRobo."""
    import numpy as np

    while True:
        req = _move_arm_executor.get_next()
        if req is None:
            continue

        if _api_core is None:
            _move_arm_executor.complete(False, "api_core not ready")
            continue

        arm = req.get("arm", "right")
        linear_move = bool(req.get("linear", False))

        if arm == "left":
            # Lazy-initialize left arm CuroboMotion on first left-arm request.
            global _left_curobo_motion
            if _left_curobo_motion is None:
                with _left_curobo_init_lock:
                    if _left_curobo_motion is None:
                        print("[MoveArm] Initializing left arm CuroboMotion (G1_omnipicker_fixed_left.yml)…")

                        def _init_left():
                            global _left_curobo_motion
                            from geniesim.app.utils.motion_gen_reacher import CuroboMotion

                            ub = _api_core.ui_builder
                            _left_curobo_motion = CuroboMotion(
                                ub.articulation,
                                ub.my_world,
                                "G1_omnipicker_fixed_left.yml",
                                ub.robot_prim_path,
                                [],
                                step=100,
                            )
                            _left_curobo_motion.set_obstacles()
                            print("[MoveArm] Left arm CuroboMotion ready.")

                        _api_core.run_on_physics_loop(_init_left)
            cm = _left_curobo_motion
        else:
            cm = _api_core.ui_builder.curoboMotion

        if cm is None:
            _move_arm_executor.complete(False, "curoboMotion not initialized; enable_curobo must be true")
            continue

        try:
            # cuRobo 内部运动学模型以 base_link 为原点，直接使用 base frame 坐标。
            pos = np.array(req["position"], dtype=np.float32)
            quat = np.array(req["quat_wxyz"], dtype=np.float32)
            quat /= np.linalg.norm(quat)
            if quat[0] < 0:
                quat = -quat

            print(f"[MoveArm] target (base frame): pos={np.round(pos, 4).tolist()}  quat={np.round(quat, 4).tolist()}")

            # Set target pose (base frame) and run motion planning on the physics thread.
            def _plan():
                from curobo.types.math import Pose as CuPose

                # Compute TRUE retract FK (not from stale prim).
                retract_cfg = cm.motion_gen.get_retract_config()
                kin_st = cm.motion_gen.kinematics.get_state(retract_cfg.view(1, -1))
                link_pose = kin_st.link_pose
                if cm.ee_link_name in link_pose:
                    rp = np.ravel(link_pose[cm.ee_link_name].to_list())
                    print(f"[cuRobo Diag] ee_link={cm.ee_link_name}  retract EE pos={np.round(rp[:3], 3).tolist()}  quat={np.round(rp[3:], 3).tolist()}")
                    # Test IK for retract pose (must succeed — if not, config is broken).
                    ik_retract = CuPose(
                        position=cm.tensor_args.to_device(rp[:3].astype(np.float32)),
                        quaternion=cm.tensor_args.to_device(rp[3:].astype(np.float32)),
                    )
                    ik_retract_res = cm.motion_gen.ik_solver.solve_single(ik_retract)
                    print(f"[cuRobo Diag] IK@retract={bool(ik_retract_res.success.item())}  (sanity check — must be True)")

                # Sync locked joints (body/head) with actual simulator state before planning.
                _api_core.ui_builder.set_locked_joint_positions()

                # Also sync cuRobo's locked joints (incl. gripper) to current physical positions.
                # The robot JSON has lock_joints=[] so set_locked_joint_positions() skips gripper joints,
                # leaving them at stale yml defaults (e.g. idx81=0.95 = partially closed).
                # This causes cuRobo to bake the default value into every trajectory frame,
                # overriding the physical open/closed state set by _do_gripper().
                _artic = _api_core.ui_builder.articulation  # type: ignore[union-attr]
                _dof_names_list = list(_artic.dof_names)  # type: ignore[union-attr]
                _dof_pos = _artic.get_joint_positions()  # type: ignore[union-attr]
                _locked_sync = {n: float(_dof_pos[_dof_names_list.index(n)])
                                for n in cm.lock_js_names if n in _dof_names_list}  # type: ignore[union-attr]
                if _locked_sync:
                    cm.update_lock_joints(_locked_sync)  # type: ignore[union-attr]

                cm.target.set_world_pose(pos, quat)
                if cm.ee_link_name in cm.target_links:
                    cm.target_links[cm.ee_link_name].set_world_pose(pos, quat)

                # linear=True: short-range move — disable graph search, trajopt only.
                # This prevents cuRobo from generating "long detour" trajectories for
                # small Cartesian moves (e.g. 2cm forward push) by forcing the optimizer
                # to stay near the current joint configuration.
                if linear_move:
                    import importlib as _il
                    _MGPC = _il.import_module("curobo.wrap.reacher.motion_gen").MotionGenPlanConfig
                    _saved_config = cm.plan_config  # type: ignore[union-attr]
                    cm.plan_config = _MGPC(  # type: ignore[union-attr]
                        enable_graph=False,       # trajopt only — no RRT* global search
                        enable_opt=True,
                        max_attempts=10,
                        enable_finetune_trajopt=True,
                        parallel_finetune=True,
                        time_dilation_factor=0.6,
                        ik_fail_return=5,
                        success_ratio=0.5,
                    )
                    print("[MoveArm] linear mode: graph search disabled, trajopt only")

                # Retry up to 3 times — cuRobo uses random restarts and can fail near workspace
                # boundaries on the first attempt but succeed on a subsequent try.
                _MAX_ATTEMPTS = 3
                for _attempt in range(_MAX_ATTEMPTS):
                    cm.caculate_ik_goal()
                    if cm.success:
                        if _attempt > 0:
                            print(f"[MoveArm] succeeded on attempt {_attempt + 1}/{_MAX_ATTEMPTS}")
                        break
                    if _attempt < _MAX_ATTEMPTS - 1:
                        print(f"[MoveArm] attempt {_attempt + 1}/{_MAX_ATTEMPTS} failed, retrying...")

                if linear_move:
                    cm.plan_config = _saved_config  # type: ignore[union-attr]  # restore original config

                if cm.success:  # type: ignore[union-attr]
                    # Strip gripper joints from the trajectory so cuRobo never overrides the
                    # gripper state set by _do_gripper() via DriveAPI.  cuRobo plans with the
                    # gripper joints for collision purposes but we don't want it to command them.
                    _jnames = cm.cmd_plan.joint_names  # type: ignore[union-attr]
                    _keep = [j for j, n in enumerate(_jnames) if "gripper" not in n]
                    if len(_keep) < len(_jnames):
                        import torch as _torch
                        from curobo.types.state import JointState as _JS
                        _kt = _torch.tensor(_keep, device=cm.cmd_plan.position.device)  # type: ignore[union-attr]
                        cm.cmd_plan = _JS(  # type: ignore[union-attr]
                            position=cm.cmd_plan.position[:, _kt],  # type: ignore[union-attr]
                            velocity=cm.cmd_plan.velocity[:, _kt],  # type: ignore[union-attr]
                            joint_names=[_jnames[j] for j in _keep],
                        )
                        cm.idx_list = [cm.idx_list[j] for j in _keep]  # type: ignore[union-attr]
                        print(f"[MoveArm] stripped {len(_jnames)-len(_keep)} gripper joints from traj, "
                              f"{len(_keep)} joints remain: {[_jnames[j] for j in _keep]}")

                if not cm.success:
                    # IK-only test: distinguishes "position unreachable" from "traj failed".
                    ik_goal = CuPose(
                        position=cm.tensor_args.to_device(pos),
                        quaternion=cm.tensor_args.to_device(quat),
                    )
                    ik_res = cm.motion_gen.ik_solver.solve_single(ik_goal)
                    ik_ok = bool(ik_res.success.item())
                    print(
                        f"[cuRobo Diag] IK-only={ik_ok}  "
                        f"({'position unreachable' if not ik_ok else 'IK ok but traj failed'})"
                        f"  status={getattr(cm, 'last_result_status', '?')}"
                    )
                    if ik_ok:
                        print(f"[cuRobo Diag] IK joints: {ik_res.js_solution.position.cpu().numpy().round(3).tolist()}")

            _api_core.run_on_physics_loop(_plan)

            if not cm.success:
                _move_arm_executor.complete(False, "cuRobo planning failed")
                continue

            # Wait for on_physics_step() to finish executing the trajectory.
            # Note: cm.reached is set when the last command frame is *dispatched*,
            # not when the robot physically arrives. Add a short settle delay so the
            # physics engine has time to drive joints to their final positions before
            # the caller (e.g. close_gripper) proceeds.
            deadline = time.time() + 60.0
            while not cm.reached:
                if time.time() > deadline:
                    _move_arm_executor.complete(False, "trajectory execution timeout")
                    break
                time.sleep(0.05)
            else:
                time.sleep(0.4)  # 等待物理引擎追上最后一帧指令，让关节稳定到位
                _move_arm_executor.complete(True)

        except Exception as exc:
            print(f"[MoveArm] worker error: {exc}")
            _move_arm_executor.complete(False, str(exc))


def _do_gripper(action: str, arm: str):
    """Set gripper joint positions. action='open'|'close', arm='left'|'right'."""
    try:
        if _api_core is None:
            return {"ok": False, "error": "api_core not ready"}

        opened_positions = getattr(_api_core, "opened_positions", None)
        finger_names = getattr(_api_core, "finger_names", None)
        if opened_positions is None or finger_names is None:
            return {"ok": False, "error": "robot not initialized"}

        f_names = finger_names.get(arm, [])
        if not f_names:
            return {"ok": False, "error": f"no finger joints for arm={arm}"}

        artic = _api_core.ui_builder.articulation
        if artic is None:
            return {"ok": False, "error": "articulation not ready"}

        dof_names = list(artic.dof_names)
        indices = [dof_names.index(n) for n in f_names if n in dof_names]
        if not indices:
            return {"ok": False, "error": "finger DOFs not found in articulation"}

        robot_base = _api_core.robot_prim_path
        stage = _api_core.ui_builder.my_world.stage

        if action == "open":
            from pxr import UsdPhysics
            from isaacsim.core.utils.types import ArticulationAction as _AA
            positions = list(opened_positions.get(arm, [2.0, 2.0]))

            def _open():
                # Restore position-drive mode, then open at velocity 40 rad/s
                for fn in f_names:
                    prim = stage.GetPrimAtPath(f"{robot_base}/joints/{fn}")
                    if prim.IsValid():
                        drv = UsdPhysics.DriveAPI.Apply(prim, "angular")
                        drv.CreateStiffnessAttr(1000.0)
                        drv.CreateMaxForceAttr(10.0)
                        drv.CreateTargetVelocityAttr(0.0)
                _pos: list = [None] * len(dof_names)
                _vel: list = [None] * len(dof_names)
                for i, p in zip(indices, positions):
                    _pos[i] = p
                    _vel[i] = 40.0
                artic.apply_action(_AA(joint_positions=_pos, joint_velocities=_vel))

            _api_core.run_on_physics_loop(_open)

        else:  # close — velocity control with force limit, same as data_collection ParallelGripper
            from pxr import UsdPhysics
            from isaacsim.core.utils.types import ArticulationAction as _AA
            closed_vels = getattr(_api_core, "closed_velocities", {}).get(arm, [-60.0, -60.0])

            def _close():
                cur_pos = artic.get_joint_positions()
                for fn, vel in zip(f_names, closed_vels):
                    if fn not in dof_names:
                        continue
                    prim = stage.GetPrimAtPath(f"{robot_base}/joints/{fn}")
                    if not prim.IsValid():
                        continue
                    # Adaptive force limit: 10 + 2*|current_position| (matches ParallelGripper)
                    fp = float(cur_pos[dof_names.index(fn)])
                    target_force = 10.0 + 2.0 * abs(fp)
                    drv = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    drv.CreateStiffnessAttr(0.0)       # velocity-drive mode
                    drv.CreateMaxForceAttr(target_force)
                    drv.CreateTargetVelocityAttr(float(vel))
                _vel: list = [None] * len(dof_names)
                for i, vel in zip(indices, closed_vels):
                    _vel[i] = float(vel)
                artic.apply_action(_AA(joint_velocities=_vel))

            _api_core.run_on_physics_loop(_close)

        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _update_frame(_robot_interface=None):
    """
    从 Isaac Sim viewport 抓帧，编码为 JPEG 写入全局缓冲。
    使用 viewport render product，不依赖机器人相机配置。
    """
    global _frame_bytes, _debug_info, _viewport_annotator
    try:
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        from omni.kit.viewport.utility import get_active_viewport

        # 懒初始化 viewport annotator
        if _viewport_annotator is None:
            viewport = get_active_viewport()
            if viewport is None:
                _debug_info["status"] = "no active viewport"
                return
            rp_path = viewport.get_render_product_path()
            _debug_info["render_product"] = rp_path
            _viewport_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            _viewport_annotator.attach([rp_path])
            # 开启 capture_on_play：仿真运行时每帧自动更新 annotator 数据
            rep.orchestrator.set_capture_on_play(True)
            _debug_info["status"] = "annotator initialized"
            return  # 第一帧下一个 tick 才可用

        raw = _viewport_annotator.get_data()
        _debug_info["raw_type"] = type(raw).__name__

        if raw is None:
            _debug_info["status"] = "get_data() returned None"
            return
        if isinstance(raw, dict):
            _debug_info["raw_dict_keys"] = list(raw.keys())
            raw = raw.get("data")
            if raw is None:
                _debug_info["status"] = "dict['data'] is None"
                return

        arr = np.asarray(raw)
        _debug_info["arr_shape"] = str(arr.shape)
        if arr.ndim < 3 or arr.size == 0:
            _debug_info["status"] = f"invalid shape {arr.shape}"
            return

        rgb = arr[..., :3]  # RGBA -> RGB
        ok, buf = cv2.imencode(".jpg", rgb[..., ::-1])  # RGB -> BGR for cv2
        if ok:
            with _frame_lock:
                _frame_bytes = buf.tobytes()
            _debug_info["status"] = f"ok, jpeg={len(_frame_bytes)}B"
        else:
            _debug_info["status"] = "imencode failed"
    except Exception as e:
        _debug_info["status"] = f"exception: {e}"
        print(f"[FrameServer] _update_frame error: {e}")


def _update_head_camera():
    """
    抓取头部相机的 RGB（JPEG）和深度图（16-bit PNG），写入全局缓冲。
    供 /head_rgb 和 /head_depth 端点使用，AnyGrasp 需要对齐的 RGB+depth。
    """
    global _head_rgb_bytes, _head_depth_bytes, _head_depth_annotator, _head_camera_info
    try:
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        from pxr import UsdGeom

        if _api_core is None:
            return
        ri = _api_core.robot_interface
        if ri is None or not hasattr(ri, "parameters"):
            return

        # 懒初始化
        if _head_depth_annotator is None:
            params = ri.parameters.get("head_camera")
            if params is None:
                return
            prim_path = params["path"]
            width = params["resolution"]["width"]
            height = params["resolution"]["height"]
            # 复用已有的 render product（不重复创建）
            rp = rep.create.render_product(prim_path, (width, height))
            _head_depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            _head_depth_annotator.attach([rp])
            rep.orchestrator.set_capture_on_play(True)  # 确保新 annotator 也纳入 capture
            # 读取相机内参
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            cam = UsdGeom.Camera(stage.GetPrimAtPath(prim_path))
            fl = cam.GetFocalLengthAttr().Get()       # mm
            ha = cam.GetHorizontalApertureAttr().Get() # mm
            va = cam.GetVerticalApertureAttr().Get()   # mm
            _head_camera_info = {
                "fx": float(fl * width / ha),
                "fy": float(fl * height / va),
                "cx": float(width / 2.0),
                "cy": float(height / 2.0),
                "width": width,
                "height": height,
                "prim_path": prim_path,
            }
            return  # 第一帧下一 tick 才可用

        # ── RGB ──
        rgb_ann = ri.annotators.get("head_camera")
        if rgb_ann is not None:
            raw = rgb_ann.get_data()
            if isinstance(raw, dict):
                raw = raw.get("data")
            if raw is not None:
                arr = np.asarray(raw)
                if arr.ndim >= 3 and arr.size > 0:
                    ok, buf = cv2.imencode(".jpg", arr[..., :3][..., ::-1])
                    if ok:
                        with _head_lock:
                            _head_rgb_bytes = buf.tobytes()

        # ── Depth ──
        raw = _head_depth_annotator.get_data()
        if isinstance(raw, dict):
            raw = raw.get("data")
        if raw is not None:
            depth_m = np.asarray(raw, dtype=np.float32)
            if depth_m.ndim >= 2 and depth_m.size > 0:
                # nan/inf 背景像素 → 0（无效深度），避免转 uint16 时产生未定义行为
                depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
                depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
                ok, buf = cv2.imencode(".png", depth_mm)
                if ok:
                    with _head_lock:
                        _head_depth_bytes = buf.tobytes()

    except Exception as e:
        print(f"[HeadCamera] _update_head_camera error: {e}")


def _get_cam_extrinsics():
    """
    从 USD Stage 读取 T_cam2base（4×4 齐次变换矩阵，相机坐标系→机器人base坐标系）。

    USD 使用行向量约定：v_world_row = v_local_row * M
    等价列向量约定：v_world = M.T @ v_local （标准机器人学惯例）
    返回 list[list[float]] 4×4，适合 JSON 序列化。
    """
    try:
        import omni.usd
        import numpy as np
        from pxr import UsdGeom, Usd

        if not _head_camera_info or "prim_path" not in _head_camera_info:
            return None

        stage = omni.usd.get_context().get_stage()
        cam_prim_path = _head_camera_info["prim_path"]
        # 从相机路径推断 base_link：第一段为机器人 namespace（如 /G1/...）
        robot_ns = cam_prim_path.lstrip("/").split("/")[0]
        base_prim_path = f"/{robot_ns}/base_link"

        time_code = Usd.TimeCode.Default()
        cam_prim = stage.GetPrimAtPath(cam_prim_path)
        base_prim = stage.GetPrimAtPath(base_prim_path)
        if not cam_prim.IsValid() or not base_prim.IsValid():
            return None

        # USD 行向量矩阵 → 列向量齐次变换矩阵（转置即可）
        T_cam_world = np.array(UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(time_code)).T
        T_base_world = np.array(UsdGeom.Xformable(base_prim).ComputeLocalToWorldTransform(time_code)).T
        T_cam2base = np.linalg.inv(T_base_world) @ T_cam_world
        # USD camera convention: Y-up, Z-backward → OpenCV (AnyGrasp): Y-down, Z-forward
        T_cam2base = T_cam2base @ np.diag([1.0, -1.0, -1.0, 1.0])
        return T_cam2base.tolist()
    except Exception as e:
        print(f"[FrameServer] _get_cam_extrinsics error: {e}")
        return None


def _start_frame_server(port: int = FRAME_SERVER_PORT):
    """在后台线程启动轻量 HTTP 服务，供 embodiedclaw agent 拉取帧。"""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            import json
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if self.path == "/action":
                result = {"ok": False, "error": "api_core not ready"}
                if _api_core is not None:
                    try:
                        action_type = body.get("type", "")
                        if action_type == "set_joints":
                            positions = body["positions"]
                            indices = body["indices"]
                            is_traj = body.get("is_trajectory", False)
                            _api_core.set_joint_positions(positions, indices, is_traj)
                            result = {"ok": True}
                        else:
                            result = {"ok": False, "error": f"unknown type: {action_type}"}
                    except Exception as e:
                        result = {"ok": False, "error": str(e)}

            elif self.path == "/run_skill":
                instruction = body.get("instruction", "")
                if _skill_executor is None:
                    result = {"ok": False, "error": "skill_executor not ready"}
                elif not instruction:
                    result = {"ok": False, "error": "instruction is empty"}
                else:
                    _skill_executor.request(instruction)
                    completed = _skill_executor.wait_for_done(timeout=120.0)
                    if completed:
                        result = {"ok": True, "result": _skill_executor.state}
                    else:
                        result = {"ok": False, "error": "timeout waiting for skill"}

            elif self.path == "/reset_scene":
                if _skill_executor is not None:
                    _skill_executor.reset()
                    result = {"ok": True}
                else:
                    result = {"ok": False, "error": "skill_executor not ready"}

            elif self.path == "/move_arm":
                if _move_arm_executor is None:
                    result = {"ok": False, "error": "move_arm not available"}
                else:
                    position = body.get("position")
                    quat_wxyz = body.get("quat_wxyz")
                    arm = body.get("arm", "right")
                    if position is None or quat_wxyz is None:
                        result = {"ok": False, "error": "missing position or quat_wxyz"}
                    else:
                        ok, err = _move_arm_executor.request(position, quat_wxyz, arm=arm)
                        result = {"ok": ok, "error": err}

            elif self.path == "/check_ik":
                # Batch Lula IK feasibility check (fast, no trajectory planning).
                # Body: {"poses": [{"position":[x,y,z], "quat_wxyz":[w,x,y,z]}, ...], "arm": "right"}
                # Returns: {"ok": true, "results": [true, false, ...]}
                if _api_core is None:
                    result = {"ok": False, "error": "api_core not ready"}
                else:
                    import numpy as _np
                    _poses = body.get("poses", [])
                    _arm = body.get("arm", "right")
                    _is_right = (_arm == "right")
                    _ik_results = []
                    _ik_errors = []

                    # Diagnostic: report solver state
                    _ub = _api_core.ui_builder
                    _arm_type = getattr(_ub, "arm_type", "MISSING")
                    _has_solver = hasattr(_ub, "kinematics_solver") and _ub.kinematics_solver is not None
                    _solver_type = type(getattr(_ub, "kinematics_solver", None)).__name__
                    print(f"[check_ik] arm={_arm} is_right={_is_right} arm_type={_arm_type} "
                          f"has_solver={_has_solver} solver_type={_solver_type} poses={len(_poses)}")

                    def _check_ik_batch():
                        # Get robot world pose for base→world conversion
                        _base_t, _base_q = _api_core.ui_builder.articulation.get_world_pose()
                        print(f"[check_ik] robot_world_pose: pos={_base_t.tolist()} quat={_base_q.tolist()}")

                        # Build base→world rotation matrix from quaternion
                        _w, _x, _y, _z = (float(_base_q[0]), float(_base_q[1]),
                                           float(_base_q[2]), float(_base_q[3]))
                        _R_b2w = _np.array([
                            [1-2*(_y*_y+_z*_z), 2*(_x*_y-_z*_w), 2*(_x*_z+_y*_w)],
                            [2*(_x*_y+_z*_w), 1-2*(_x*_x+_z*_z), 2*(_y*_z-_x*_w)],
                            [2*(_x*_z-_y*_w), 2*(_y*_z+_x*_w), 1-2*(_x*_x+_y*_y)],
                        ])

                        for _i, _pose in enumerate(_poses):
                            _pos_base = _np.array(_pose["position"], dtype=_np.float64)
                            _quat_base = _np.array(_pose["quat_wxyz"], dtype=_np.float64)

                            # Convert position: base frame → world frame
                            _pos_world = (_R_b2w @ _pos_base + _base_t).astype(_np.float32)

                            # Convert orientation: q_world = q_base2world * q_target_base
                            _bw, _bx, _by, _bz = _w, _x, _y, _z
                            _tw, _tx, _ty, _tz = _quat_base
                            _quat_world = _np.array([
                                _bw*_tw - _bx*_tx - _by*_ty - _bz*_tz,
                                _bw*_tx + _bx*_tw + _by*_tz - _bz*_ty,
                                _bw*_ty - _bx*_tz + _by*_tw + _bz*_tx,
                                _bw*_tz + _bx*_ty - _by*_tx + _bz*_tw,
                            ], dtype=_np.float32)

                            try:
                                _ok, _actions = _api_core.ui_builder._get_ik_status(
                                    _pos_world, _quat_world, _is_right)
                                _ik_results.append(bool(_ok))
                                if not _ok:
                                    print(f"[check_ik] [{_i}] FAIL base={_pos_base.tolist()} "
                                          f"world={_pos_world.tolist()}")
                                else:
                                    print(f"[check_ik] [{_i}] OK   base={_pos_base.tolist()} "
                                          f"world={_pos_world.tolist()}")
                            except Exception as _ex:
                                _ik_results.append(False)
                                _err_msg = f"{type(_ex).__name__}: {_ex}"
                                _ik_errors.append(_err_msg)
                                print(f"[check_ik] [{_i}] EXCEPTION: {_err_msg}")

                    try:
                        _api_core.run_on_physics_loop(_check_ik_batch)
                        result = {"ok": True, "results": _ik_results,
                                  "arm_type": _arm_type, "has_solver": _has_solver,
                                  "solver_type": _solver_type}
                        if _ik_errors:
                            result["errors"] = _ik_errors
                    except Exception as _e:
                        result = {"ok": False, "error": str(_e)}

            elif self.path == "/gripper":
                action = body.get("action", "open")
                arm = body.get("arm", "right")
                result = _do_gripper(action, arm)

            else:
                self.send_response(404)
                self.end_headers()
                return

            resp = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp)

        def do_GET(self):
            if self.path == "/frame":
                with _frame_lock:
                    data = _frame_bytes
                if data:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"no frame yet")
            elif self.path == "/healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            elif self.path == "/joints":
                import json
                result = {"ok": False, "error": "api_core not ready"}
                if _api_core is not None:
                    try:
                        jd = _api_core.get_joint_state_dict()
                        result = {"ok": True, "joints": jd}
                    except Exception as e:
                        result = {"ok": False, "error": str(e)}
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/head_rgb":
                with _head_lock:
                    data = _head_rgb_bytes
                if data:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"head rgb not ready")
            elif self.path == "/head_depth":
                with _head_lock:
                    data = _head_depth_bytes
                if data:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"head depth not ready")
            elif self.path == "/intrinsics":
                import json
                if _head_camera_info:
                    body = json.dumps(_head_camera_info).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"intrinsics not ready")
            elif self.path == "/cam_extrinsics":
                import json
                mat = _get_cam_extrinsics()
                if mat is not None:
                    body = json.dumps({"matrix": mat}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"cam extrinsics not ready")
            elif self.path == "/task_status":
                import json
                state = _skill_executor.state if _skill_executor is not None else "unavailable"
                body = json.dumps({"state": state}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/debug":
                import json
                info = dict(_debug_info)
                info["frame_bytes_len"] = len(_frame_bytes)
                body = json.dumps(info, indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass  # 静默日志

    server = HTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[FrameServer] http://localhost:{port}/frame")
# ─────────────────────────────────────────────────────────────────────────────

from isaacsim.core.utils import extensions

if cfg.app.enable_ros:
    extensions.enable_extension("isaacsim.ros2.bridge")


def wait_rclpy(timeout=10, tick=0.1):
    """Block until rclpy can be imported, or raise after <timeout> seconds."""
    start = time.time()
    while True:
        try:
            import rclpy

            return rclpy
        except ModuleNotFoundError:
            if time.time() - start > timeout:
                raise RuntimeError("rclpy still not available")
            time.sleep(tick)


if cfg.app.enable_ros:
    rclpy = wait_rclpy()
    rclpy.init()
else:
    rclpy = None

from isaacsim.core.api import World
from geniesim.app.controllers import APICore
from geniesim.app.task_manager import TaskManager
from geniesim.app.workflow.ui_builder import UIBuilder


def main():
    """Main function."""

    world = World(
        stage_units_in_meters=1,
        physics_dt=1.0 / cfg.app.physics_step,
        rendering_dt=1.0 / cfg.app.rendering_step,
    )
    if cfg.app.enable_gpu_dynamics:
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
        world._physics_context.enable_gpu_dynamics(flag=True)
        world._physics_context.enable_ccd(flag=True)
    ui_builder = UIBuilder(world=world)
    task_manager = TaskManager(
        api_core=APICore(ui_builder=ui_builder, config=cfg),
        benchmark_config=cfg.benchmark,
    )

    def callback_physics(step_size):
        global _frame_count, _last_time
        _frame_count += 1
        now = time.time()
        elapsed = now - _last_time
        if elapsed >= 1.0:
            # hz = _frame_count / elapsed
            # print(f"[Physics Callback] {hz:.2f} Hz")
            _frame_count = 0
            _last_time = now

        if task_manager:
            task_manager.api_core.physics_step()
            task_manager.api_core.on_ros_tick(step_size)
        ui_builder.on_physics_step(step_size)  # steps right arm cuRobo trajectory
        if _left_curobo_motion is not None:
            _left_curobo_motion.on_physics_step()  # steps left arm cuRobo trajectory

    if cfg.app.enable_embodied_http:
        global _api_core, _skill_executor, _move_arm_executor
        _api_core = task_manager.api_core
        _skill_executor = SkillExecutor()
        _api_core.skill_executor = _skill_executor
        _move_arm_executor = MoveArmExecutor()
        threading.Thread(target=_move_arm_worker, daemon=True, name="move_arm_worker").start()

    ui_builder.my_world.add_physics_callback("on_physics", callback_fn=callback_physics)
    task_manager.start()

    if cfg.app.enable_embodied_http:
        _start_frame_server()

    step = 0
    while simulation_app.is_running():
        ui_builder.my_world.step(render=True)
        task_manager.api_core.render_step()
        if cfg.app.enable_embodied_http and step % 15 == 0:  # ~2Hz frame update
            import omni.replicator.core as _rep
            _rep.orchestrator.step(pause_timeline=False)
            _update_frame(task_manager.api_core.robot_interface)
            _update_head_camera()

        step += 1  # 每帧递增，确保 step % 15 按真实渲染帧节流

        if task_manager.api_core.exit:
            task_manager.api_core.post_process()
            break

        if not ui_builder.my_world.is_playing():
            if step % 100 == 0:
                print("**** simulation paused ****")
            continue

    simulation_app.close()


if __name__ == "__main__":
    main()

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
简单键盘遥操作脚本 - 直接通过gRPC控制机器人末端执行器
无需 ROS2 / pxr / cuRobo 等依赖，只需 grpcio (Isaac Sim 内置)

用法:
  /home/qkyjy/IsaacLab-main/_isaac_sim/python.sh source/geniesim/teleop/keyboard_teleop_simple.py

前提条件:
  1. GenieSim 仿真已启动: /isaac-sim/python.sh source/geniesim/app/app.py --config ...
  2. gRPC 服务器运行在 localhost:50051
  3. 确保已 conda deactivate (避免 Python 版本冲突)

按键说明:
  w / s     : Y轴 前进 / 后退
  a / d     : X轴 左移 / 右移
  q / e     : Z轴 上升 / 下降
  c         : 关闭夹爪 (抓取)
  v         : 打开夹爪 (释放)
  TAB       : 切换左臂 / 右臂
  p         : 打印当前末端位姿
  + / -     : 增大 / 减小步长
  ESC       : 退出
"""

import os
import sys
import tty
import termios

# ---- 路径设置: 让 Python 能找到 common.aimdk.protocol.* 的 proto stubs ----
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_this_dir, "../../.."))
_data_collection_dir = os.path.join(_repo_root, "source/data_collection")
if _data_collection_dir not in sys.path:
    sys.path.insert(0, _data_collection_dir)

import grpc

from common.aimdk.protocol.hal.arm import arm_pb2, arm_pb2_grpc
from common.aimdk.protocol.hal.joint import joint_channel_pb2, joint_channel_pb2_grpc
from common.aimdk.protocol.sim import sim_gripper_service_pb2, sim_gripper_service_pb2_grpc

SERVER_ADDR = "localhost:50051"
DEFAULT_STEP = 0.01  # 单次移动步长 (米)


class KeyboardTeleop:
    def __init__(self):
        print(f"正在连接 gRPC 服务器: {SERVER_ADDR} ...")
        self.channel = grpc.insecure_channel(
            SERVER_ADDR,
            options=[("grpc.max_receive_message_length", 16094304)],
        )
        try:
            grpc.channel_ready_future(self.channel).result(timeout=10)
        except grpc.FutureTimeoutError:
            print(f"错误: 无法连接到 {SERVER_ADDR}")
            print("请确认 GenieSim 仿真已启动并且 gRPC 服务端口正常。")
            sys.exit(1)

        self.arm_stub = arm_pb2_grpc.ArmControlServiceStub(self.channel)
        self.joint_stub = joint_channel_pb2_grpc.JointControlServiceStub(self.channel)
        self.gripper_stub = sim_gripper_service_pb2_grpc.SimGripperServiceStub(self.channel)

        self.is_right = True  # True = 右臂,  False = 左臂
        self.step = DEFAULT_STEP

    def get_ee_pose(self):
        """获取当前末端执行器位姿，返回 (pos [x,y,z], quat [w,x,y,z])"""
        req = joint_channel_pb2.GetEEPoseReq()
        req.is_right = self.is_right
        resp = self.joint_stub.get_ee_pose(req)
        pos = [resp.ee_pose.position.x, resp.ee_pose.position.y, resp.ee_pose.position.z]
        # 注意: SE3RpyPose 的 rpy 字段实际存储的是 wxyz 四元数 (rw=w, rx=x, ry=y, rz=z)
        quat = [resp.ee_pose.rpy.rw, resp.ee_pose.rpy.rx, resp.ee_pose.rpy.ry, resp.ee_pose.rpy.rz]
        return pos, quat

    def moveto(self, pos, quat):
        """发送末端位姿目标, 阻塞直到运动完成"""
        arm_name = "right" if self.is_right else "left"
        req = arm_pb2.LinearMoveReq()
        req.robot_name = arm_name
        req.pose.position.x, req.pose.position.y, req.pose.position.z = pos
        req.pose.rpy.rw, req.pose.rpy.rx, req.pose.rpy.ry, req.pose.rpy.rz = quat
        req.is_block = True
        req.ee_interpolation = False
        req.distance_frame = 0.0008
        req.goal_offset[:] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        req.offset_and_constraint_in_goal_frame = True
        req.motion_run_ratio = 1.0
        req.gripper_action_timing = "{}"
        req.from_current_pose = False
        resp = self.arm_stub.linear_move(req)
        return resp

    def set_gripper(self, command):
        """控制夹爪: command = 'open' 或 'close'"""
        req = sim_gripper_service_pb2.SetGripperStateReq()
        req.gripper_command = command
        req.is_right = self.is_right
        req.opened_width = 0.08 if command == "open" else 0.00
        self.gripper_stub.set_gripper_state(req)

    def _arm_label(self):
        return "右臂" if self.is_right else "左臂"

    def run(self):
        # 验证连接并打印初始位姿
        try:
            pos, quat = self.get_ee_pose()
            print(f"连接成功! {self._arm_label()} 当前位置: "
                  f"x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")
        except Exception as e:
            print(f"获取末端位姿失败: {e}")
            print("仿真可能尚未完全初始化，请稍后重试。")
            self.channel.close()
            return

        print()
        print("=== GenieSim 键盘遥操作 ===")
        print("  w/s    : Y轴 前进/后退")
        print("  a/d    : X轴 左移/右移")
        print("  q/e    : Z轴 上升/下降")
        print("  c      : 关闭夹爪 (抓取)")
        print("  v      : 打开夹爪 (释放)")
        print("  TAB    : 切换左/右臂")
        print("  p      : 打印当前末端位姿")
        print("  +/-    : 增大/减小步长 (当前: {:.3f}m)".format(self.step))
        print("  ESC    : 退出")
        print("=" * 35)
        print("(每次按键后等待运动完成再按下一个键)\n")

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)

                # ESC 或 Ctrl+C 退出
                if ch in ("\x1b", "\x03"):
                    print("\r\n退出遥操作。")
                    break

                # 方向键前缀 \x1b[ → 忽略后续两个字节
                if ch == "\x1b":
                    sys.stdin.read(2)
                    continue

                # 切换臂
                if ch == "\t":
                    self.is_right = not self.is_right
                    try:
                        pos, quat = self.get_ee_pose()
                        print(f"\r切换到{self._arm_label()}: "
                              f"x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}    ")
                    except Exception as e:
                        print(f"\r切换到{self._arm_label()} (获取位姿失败: {e})")
                    continue

                # 步长调节
                if ch == "+":
                    self.step = min(self.step + 0.005, 0.10)
                    print(f"\r步长: {self.step:.3f}m    ", end="", flush=True)
                    continue
                if ch == "-":
                    self.step = max(self.step - 0.005, 0.005)
                    print(f"\r步长: {self.step:.3f}m    ", end="", flush=True)
                    continue

                # 打印位姿
                if ch == "p":
                    try:
                        pos, quat = self.get_ee_pose()
                        print(f"\r{self._arm_label()} pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})  "
                              f"quat=(w={quat[0]:.3f}, x={quat[1]:.3f}, y={quat[2]:.3f}, z={quat[3]:.3f})    ")
                    except Exception as e:
                        print(f"\r获取位姿失败: {e}")
                    continue

                # 夹爪控制
                if ch == "c":
                    print(f"\r关闭{self._arm_label()}夹爪...    ", end="", flush=True)
                    try:
                        self.set_gripper("close")
                        print(f"\r{self._arm_label()}夹爪已关闭    ", end="", flush=True)
                    except Exception as e:
                        print(f"\r夹爪控制失败: {e}")
                    continue

                if ch == "v":
                    print(f"\r打开{self._arm_label()}夹爪...    ", end="", flush=True)
                    try:
                        self.set_gripper("open")
                        print(f"\r{self._arm_label()}夹爪已打开    ", end="", flush=True)
                    except Exception as e:
                        print(f"\r夹爪控制失败: {e}")
                    continue

                # 末端位移控制
                delta = {
                    "w": (0, +self.step, 0),
                    "s": (0, -self.step, 0),
                    "a": (-self.step, 0, 0),
                    "d": (+self.step, 0, 0),
                    "q": (0, 0, +self.step),
                    "e": (0, 0, -self.step),
                }.get(ch)

                if delta is None:
                    continue  # 未知按键，忽略

                try:
                    pos, quat = self.get_ee_pose()
                except Exception as e:
                    print(f"\r获取位姿失败: {e}")
                    continue

                new_pos = [pos[0] + delta[0], pos[1] + delta[1], pos[2] + delta[2]]
                key_labels = {
                    "w": "+Y(前)", "s": "-Y(后)",
                    "a": "-X(左)", "d": "+X(右)",
                    "q": "+Z(上)", "e": "-Z(下)",
                }
                print(
                    f"\r{self._arm_label()} {key_labels[ch]}  "
                    f"→ ({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})  发送中...",
                    end="",
                    flush=True,
                )
                try:
                    resp = self.moveto(new_pos, quat)
                    print(
                        f"\r{self._arm_label()} {key_labels[ch]}  "
                        f"→ ({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})  完成    ",
                        end="",
                        flush=True,
                    )
                    if resp.errmsg and resp.errmsg not in ("True", ""):
                        print(f"\n  (服务器返回: {resp.errmsg})")
                except Exception as ex:
                    print(f"\r移动异常: {ex}")

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self.channel.close()
            print()


if __name__ == "__main__":
    teleop = KeyboardTeleop()
    teleop.run()

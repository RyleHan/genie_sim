"""
chem_waste_sorting 场景初始化脚本
用法 (先确保 data_collector_server.py 已启动):
  conda deactivate
  cd /home/qkyjy/embodied/genie_sim/source/data_collection
  SIM_ASSETS=/home/qkyjy/embodied/sim_assets \\
    /home/qkyjy/IsaacLab-main/_isaac_sim/python.sh \\
    scripts/init_scene_chem_waste.py

功能:
  1. 调用 init_robot gRPC → 加载 teleop_scene.usda (工厂房间 + 工作台) + G1 机器人
  2. 调用 add_object gRPC → 加载废弃物 (rigid body, 可被抓取)
  3. 打印就绪提示后退出 (服务器继续运行, 再运行 keyboard_teleop_simple.py)

不依赖 pinocchio / cuRobo / ROS2
"""

import os
import sys
import time

# ---- Path setup: find data_collection root ----
_this_dir = os.path.dirname(os.path.abspath(__file__))
_dc_root = os.path.dirname(_this_dir)  # source/data_collection/
if _dc_root not in sys.path:
    sys.path.insert(0, _dc_root)

import grpc

from common.aimdk.protocol.hal.joint import joint_channel_pb2
from common.aimdk.protocol.sim import (
    sim_object_service_pb2,
    sim_object_service_pb2_grpc,
    sim_observation_service_pb2,
    sim_observation_service_pb2_grpc,
)

SERVER_ADDR = "localhost:50051"

# Absolute path to the combined static scene USD
_SCENE_USD = os.path.join(
    os.path.dirname(_this_dir),
    "tasks/chem_waste_sorting_g1/teleop_scene.usda",
)

# G1 initial arm joint positions (from sort_fruit/g1 task template)
G1_INIT_ARM_POSE = {
    "idx01_body_joint1": 0.42,
    "idx02_body_joint2": 0.8726646991494927,
    "idx11_head_joint1": 0.0,
    "idx12_head_joint2": 0.4363296423945109,
    "idx21_arm_l_joint1": -1.0743441581726074,
    "idx22_arm_l_joint2": 0.6110641360282898,
    "idx23_arm_l_joint3": 0.2795839011669159,
    "idx24_arm_l_joint4": -1.2839884757995603,
    "idx25_arm_l_joint5": 0.7304918766021729,
    "idx26_arm_l_joint6": 1.4953428506851196,
    "idx27_arm_l_joint7": -0.18755260109901428,
    "idx61_arm_r_joint1": 1.0743441581726074,
    "idx62_arm_r_joint2": -0.6110466122627258,
    "idx63_arm_r_joint3": -0.2795839011669159,
    "idx64_arm_r_joint4": 1.283866286277771,
    "idx65_arm_r_joint5": -0.7303872108459473,
    "idx66_arm_r_joint6": -1.4952731132507324,
    "idx67_arm_r_joint7": 0.18760496377944944,
    "idx41_gripper_l_outer_joint1": 1.0,
    "idx81_gripper_r_outer_joint1": 1.0,
}

# Waste objects to add as rigid bodies (can be grasped)
# Format: (prim_path, usd_path, (x, y, z))
_WASTE_OBJECTS = [
    (
        "/World/Objects/waste_pet_bottle_nongfu",
        "/home/qkyjy/data/OurAssets/Part/WaterBottle_NongfuSpring/M_WaterBottle_NongfuSpring.usd",
        (-0.05, 0.20, 0.70),
    ),
    (
        "/World/Objects/waste_pet_bottle_estbon",
        "/home/qkyjy/data/OurAssets/Part/WaterBottle_estbon/M_WaterBottle_estbon.usd",
        (0.03, 0.22, 0.70),
    ),
    (
        "/World/Objects/waste_pe_shampoo",
        "/home/qkyjy/data/OurAssets/Part/ShampooBottle500L_Blue/M_ShampooBottle500L_Blue.usd",
        (-0.05, 0.28, 0.70),
    ),
    (
        "/World/Objects/waste_pp_lunchbox",
        "/home/qkyjy/data/OurAssets/Part/PPLunchBox _Black/M_PPLunchBox_Black.usd",
        (0.02, 0.30, 0.70),
    ),
    (
        "/World/Objects/waste_pvc_pipe",
        "/home/qkyjy/data/OurAssets/Part/PVCWaterPipe4/M_PVCWaterPipe4.usd",
        (-0.04, 0.24, 0.70),
    ),
]


def connect(addr: str, timeout: int = 30) -> grpc.Channel:
    print(f"Connecting to gRPC server: {addr} ...")
    channel = grpc.insecure_channel(
        addr,
        options=[("grpc.max_receive_message_length", 16094304)],
    )
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        print(f"ERROR: Cannot connect to {addr}")
        print("Make sure data_collector_server.py is running first.")
        sys.exit(1)
    print("Connected.")
    return channel


def init_robot(channel: grpc.Channel) -> None:
    stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(channel)
    req = sim_observation_service_pb2.InitRobotReq()

    req.robot_cfg_file = "G1_omnipicker_fixed_dual.json"
    req.robot_usd_path = ""          # resolved from robot_cfg_file
    req.scene_usd_path = _SCENE_USD  # absolute path — os.path.join ignores SIM_ASSETS prefix

    # Robot stands in front of workbench (workbench is at y=0.5)
    req.robot_pose.position.x = -0.76
    req.robot_pose.position.y = 0.0
    req.robot_pose.position.z = 0.0
    req.robot_pose.rpy.rw = 1.0
    req.robot_pose.rpy.rx = 0.0
    req.robot_pose.rpy.ry = 0.0
    req.robot_pose.rpy.rz = 0.0

    req.stand_type = "cylinder"
    req.stand_size_x = 0.1
    req.stand_size_y = 0.1

    for name, pos in G1_INIT_ARM_POSE.items():
        if pos is not None:
            jc = joint_channel_pb2.JointCommand()
            jc.name = name
            jc.position = float(pos)
            req.joint_cmd.append(jc)

    print(f"Loading scene: {_SCENE_USD}")
    print("Loading G1 robot... (this may take 30-60s)")
    resp = stub.init_robot(req)
    print(f"init_robot response: {resp.msg}")


def add_waste_objects(channel: grpc.Channel) -> None:
    stub = sim_object_service_pb2_grpc.SimObjectServiceStub(channel)

    for prim_path, usd_path, (px, py, pz) in _WASTE_OBJECTS:
        obj_name = prim_path.split("/")[-1]
        req = sim_object_service_pb2.AddObjectReq()
        req.usd_path = usd_path       # absolute path works with os.path.join
        req.prim_path = prim_path
        req.label_name = obj_name
        req.object_color.r = 1.0
        req.object_color.g = 1.0
        req.object_color.b = 1.0
        req.object_pose.position.x = px
        req.object_pose.position.y = py
        req.object_pose.position.z = pz
        req.object_pose.rpy.rw = 1.0
        req.object_pose.rpy.rx = 0.0
        req.object_pose.rpy.ry = 0.0
        req.object_pose.rpy.rz = 0.0
        req.object_scale.x = 1.0
        req.object_scale.y = 1.0
        req.object_scale.z = 1.0
        req.object_material = ""
        req.object_mass = 0.05        # ~50g per item
        req.add_particle = False
        req.static_friction = 0.6
        req.dynamic_friction = 0.4
        req.particle_position.x = 0.0
        req.particle_position.y = 0.0
        req.particle_position.z = 0.0
        req.particle_scale.x = 0.1
        req.particle_scale.y = 0.1
        req.particle_scale.z = 0.1
        req.particle_color.r = 1.0
        req.particle_color.g = 1.0
        req.particle_color.b = 1.0
        req.add_rigid_body = True     # must be True for grasping
        req.model_type = "convexDecomposition"

        print(f"  Adding {obj_name} at ({px:.2f}, {py:.2f}, {pz:.2f}) ...")
        resp = stub.add_object(req)
        print(f"    -> {resp}")


if __name__ == "__main__":
    channel = connect(SERVER_ADDR)

    init_robot(channel)

    print("\nAdding waste objects as rigid bodies...")
    add_waste_objects(channel)

    channel.close()

    print("\n========================================")
    print("Scene ready! Now run keyboard teleop:")
    print("  cd /home/qkyjy/embodied/genie_sim")
    print("  /home/qkyjy/IsaacLab-main/_isaac_sim/python.sh \\")
    print("    source/geniesim/teleop/keyboard_teleop_simple.py")
    print("========================================")

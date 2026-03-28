# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, time
import json
from .base import BasePolicy

from scipy.spatial.transform import Rotation as R


from geniesim.plugins.logger import Logger
import geniesim.utils.system_utils as system_utils
from geniesim.utils.comm.websocket_client import WebsocketClientPolicy
from geniesim.utils.comm.http_client import HttpClientPolicy
from collections import deque
import numpy as np
import geniesim.utils.system_utils as system_utils

logger = Logger()  # Create singleton instance

ROOT_DIR = os.environ.get("SIM_REPO_ROOT")
GMA_DIR = os.environ.get("SIM_GMA_PATH", os.path.join(ROOT_DIR, "../gma"))

TIMEOUT_SEC = 30


class Go1Policy(BasePolicy):
    def __init__(self, task_name, host_ip, port):
        super().__init__()
        self.ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(time.time()))
        self.task_name = task_name
        self.initialized = False
        # pi
        # self.policy = WebsocketClientPolicy(host=host_ip, port=port)
        # go1
        self.policy = HttpClientPolicy(host=host_ip, port=port)
        self.debug = False
        self.infer_cnt = 0

    def get_payload(self, obs, task_instruction):
        states = np.zeros((32,))
        states[:16] = obs["states"]
        #  pi05
        # payload = {
        #     "state": states,
        #     "images": {
        #         "top_head": np.transpose(obs["images"]["head"], (2, 0, 1)),
        #         "hand_left": np.transpose(obs["images"]["left_hand"], (2, 0, 1)),
        #         "hand_right": np.transpose(obs["images"]["right_hand"], (2, 0, 1)),
        #     },
        #     "prompt": task_instruction,
        # }

        # go1: 模型训练时 state 顺序为 [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
        # pi_env 的 obs["states"] 顺序为 [left_arm(7), right_arm(7), left_gripper(1), right_gripper(1)]
        # 需重排为模型期望顺序，否则右臂输入错位会导致右臂乱甩
        raw = obs["states"][:16]
        states_go1 = np.zeros((1, 16), dtype=np.float32)
        states_go1[0, 0:7] = raw[0:7]    # left_arm
        states_go1[0, 7] = raw[14]       # left_gripper
        states_go1[0, 8:15] = raw[7:14]   # right_arm
        states_go1[0, 15] = raw[15]       # right_gripper
        payload = {
            "top": obs["images"]["head"],
            "right": obs["images"]["right_hand"],
            "left": obs["images"]["left_hand"],
            "instruction": task_instruction,
            "state": states_go1,
            "ctrl_freqs": np.array([30]),
        }
        logger.info(f"top_shape: {payload['top'].shape}")
        return payload

    def reset(self):
        self.action_buffer.clear()

    def infer(self, payload):
        try:
            result = self.policy.infer(payload)
            logger.info(f"Inference result: {result}")
            numpy_result = np.array(result.json(), dtype=np.float32)
            logger.info(f"Result data: {numpy_result}")
            logger.info(f"Result data shape: {numpy_result.shape}")
            # pi
            # for action in result["actions"]:
            #     self.action_buffer.append(action)
            # go1
            for action in numpy_result:
                self.action_buffer.append(action)
            return True
        except Exception as e:
            logger.warning(f"Model inference failed: {str(e)}")
            time.sleep(1)
            return False

    def act(self, observation, **kwargs):
        if len(self.action_buffer) == 0:
            logger.info("policy.at call model infer")
            task_instruction = kwargs.get("task_instruction", "")
            logger.info(f"\nInstruction: {task_instruction}\n")
            payload = self.get_payload(observation, task_instruction)
            infer_success = self.infer(payload)
            logger.info(f"Inference success: {infer_success}")
            infer_start = time.time()

            while not infer_success:
                infer_success = self.infer(payload)
                if time.time() - infer_start > TIMEOUT_SEC:
                    logger.error("Model inference timeout, please check if the policy server is running normally")
                    os.system("pkill -9 -f app.py")
            self.infer_cnt += 1

        return self.action_buffer.popleft()

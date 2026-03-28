#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""
GO-1 WebSocket Inference Server
Fully compatible with unmodified pipolicy.py (WebSocket + msgpack + numpy).

Usage:
    python Zero/ws_server.py \
        --model-path /path/to/checkpoint \
        --data-stats-path /path/to/dataset_stats.json      # required only if model.config.norm=True
        --port 8999
"""

import asyncio
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import msgpack
import numpy as np
import torch
import websockets
from PIL import Image
from transformers import AutoTokenizer

from go1.internvl.model.go1 import GO1Model, GO1ModelConfig
from go1.internvl.train.constants import IMG_END_TOKEN
from go1.internvl.train.dataset import build_transform, dynamic_preprocess, preprocess_internvl2_5


# ── msgpack + numpy (mirrors geniesim/utils/msgpack_numpy.py exactly) ────────

def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    return obj

def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"],
                          dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    return obj

packb   = lambda d: msgpack.packb(d, default=_pack_array)
unpackb = lambda d: msgpack.unpackb(d, object_hook=_unpack_array)


# ── GO-1 model wrapper ────────────────────────────────────────────────────────

class GO1InferServer:
    """
    Loads and runs the GO-1 model.
    Accepts observations in pipolicy.py format, returns action chunk.
    """

    def __init__(self, model_path: Union[str, Path],
                 data_stats_path: Optional[Union[str, Path]] = None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading GO-1 from {model_path} on {self.device}")

        # ── model config ──────────────────────────────────────────────────────
        self.config = GO1ModelConfig.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=False,
        )
        if not hasattr(self.config, "initializer_range"):
            self.config.initializer_range = 0.02

        self.image_size       = self.config.force_image_size
        self.num_image_token  = int(
            (self.image_size // self.config.vision_config.patch_size) ** 2
            * self.config.downsample_ratio ** 2
        )
        self.dynamic_image_size = self.config.dynamic_image_size

        # ── model weights ─────────────────────────────────────────────────────
        self.model = GO1Model.from_pretrained(model_path, config=self.config)
        self.model.to(torch.bfloat16).to(self.device).eval()

        # ── preprocessing ─────────────────────────────────────────────────────
        self.img_transform = build_transform(
            is_train=False,
            input_size=self.image_size,
            pad2square=self.config.pad2square,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_eos_token=False, trust_remote_code=True, use_fast=False
        )

        # ── normalization stats ───────────────────────────────────────────────
        self.norm = getattr(self.config, "norm", False)
        if self.norm:
            assert data_stats_path is not None, \
                "data_stats_path required when model config has norm=True"
            with open(data_stats_path, "r") as f:
                raw = json.load(f)
            self.state_mean = torch.tensor(raw["state"]["mean"], dtype=torch.float32)
            self.state_std  = torch.tensor(raw["state"]["std"],  dtype=torch.float32)
            self.action_mean = torch.tensor(raw["action"]["mean"], dtype=torch.float32)
            self.action_std  = torch.tensor(raw["action"]["std"],  dtype=torch.float32)
            logging.info("Normalization enabled, stats loaded.")
        else:
            logging.info("Normalization disabled (model config norm=False).")

        logging.info("GO-1 model ready.")

    # ── input adaptation: pipolicy format → GO-1 internal format ─────────────

    def _adapt_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        pipolicy.py sends:
            state:  np.ndarray (32,) float
                    order: [left_arm(7), right_arm(7), left_gripper(1), right_gripper(1), zeros...]
            images: {
                "top_head":   np.ndarray (C, H, W) uint8,   ← transposed by pipolicy
                "hand_left":  np.ndarray (C, H, W) uint8,
                "hand_right": np.ndarray (C, H, W) uint8,
            }
            prompt: str

        GO-1 model expects:
            cam_head_color:       PIL.Image (H, W, C)
            cam_hand_right_color: PIL.Image (H, W, C)
            cam_hand_left_color:  PIL.Image (H, W, C)
            state:  torch.Tensor (1, 16) bfloat16
                    order: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
            ctrl_freqs: torch.Tensor (1,)
            final_prompt: str
        """
        # 1. Images: (C,H,W) uint8 → (H,W,C) → PIL
        def chw_to_pil(arr: np.ndarray) -> Image.Image:
            if arr.shape[0] == 3:                    # (C,H,W) → (H,W,C)
                arr = np.transpose(arr, (1, 2, 0))
            return Image.fromarray(arr.astype(np.uint8))

        imgs = obs["images"]
        adapted = {
            "cam_head_color":       chw_to_pil(imgs["top_head"]),
            "cam_hand_right_color": chw_to_pil(imgs["hand_right"]),
            "cam_hand_left_color":  chw_to_pil(imgs["hand_left"]),
        }

        # 2. State: reorder + reshape  (32,) → (1, 16)
        #    pipolicy order: [L_arm(0:7), R_arm(7:14), L_gripper(14), R_gripper(15)]
        #    GO-1 order:     [L_arm(0:7), L_gripper(7), R_arm(8:15),  R_gripper(15)]
        raw = obs["state"][:16].astype(np.float32)
        state = np.zeros((1, 16), dtype=np.float32)
        state[0, 0:7]  = raw[0:7]    # left arm
        state[0, 7]    = raw[14]     # left gripper
        state[0, 8:15] = raw[7:14]   # right arm
        state[0, 15]   = raw[15]     # right gripper
        adapted["state"] = torch.from_numpy(state)

        # 3. ctrl_freqs: control frequency in Hz
        adapted["ctrl_freqs"] = torch.tensor([30.0], dtype=torch.float32)

        # 4. Prompt
        prompt = obs.get("prompt", "")
        adapted["final_prompt"] = f"What action should the robot take to {prompt}?"

        return adapted

    # ── tokenize + vision encode ──────────────────────────────────────────────

    def _build_model_inputs(self, adapted: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        cam_keys = ["cam_head_color", "cam_hand_right_color", "cam_hand_left_color"]
        inputs = multi_image_get_item(
            raw_target=adapted,
            img_transform=self.img_transform,
            text_tokenizer=self.tokenizer,
            num_image_token=self.num_image_token,
            cam_keys=cam_keys,
            dynamic_image_size=self.dynamic_image_size,
            use_thumbnail=self.config.use_thumbnail,
            min_dynamic_patch=self.config.min_dynamic_patch,
            max_dynamic_patch=self.config.max_dynamic_patch,
            image_size=self.image_size,
        )
        inputs["state"]     = adapted["state"]
        inputs["ctrl_freqs"] = adapted["ctrl_freqs"]
        return inputs

    # ── forward pass ─────────────────────────────────────────────────────────

    def infer(self, obs: Dict[str, Any]) -> List[np.ndarray]:
        """
        Full pipeline: obs (pipolicy format) → action chunk (list of np arrays).
        """
        adapted = self._adapt_obs(obs)
        inputs  = self._build_model_inputs(adapted)

        state     = inputs["state"]
        ctrl_freqs = inputs["ctrl_freqs"]

        if self.norm:
            state = (state - self.state_mean) / (self.state_std + 1e-6)

        t0 = time.time()
        with torch.no_grad():
            action_out = self.model(
                pixel_values  = inputs["pixel_values"].to(dtype=torch.bfloat16, device=self.device),
                input_ids     = inputs["input_ids"].to(device=self.device).unsqueeze(0),
                attention_mask= inputs["attention_mask"].to(device=self.device).unsqueeze(0),
                position_ids  = inputs["position_ids"].to(device=self.device).unsqueeze(0),
                image_flags   = inputs["image_flags"].to(device=self.device),
                state         = state.to(dtype=torch.bfloat16, device=self.device).unsqueeze(0),
                ctrl_freqs    = ctrl_freqs.to(dtype=torch.bfloat16, device=self.device).unsqueeze(0),
            )
        logging.info(f"Inference: {(time.time()-t0)*1000:.1f} ms")

        actions = action_out[1][0].float().cpu()   # (horizon, action_dim)
        if self.norm:
            actions = actions * self.action_std + self.action_mean

        actions_np = actions.numpy()
        logging.info(f"[DIAG] raw GO-1 output[0]: L_arm={actions_np[0,0:7].tolist()}  L_grip={actions_np[0,7]:.3f}  R_arm={actions_np[0,8:15].tolist()}  R_grip={actions_np[0,15]:.3f}")

        # Reorder GO-1 output → PiEnv expected order
        # GO-1 trains with: [L_arm(0:7), L_gripper(7), R_arm(8:15), R_gripper(15)]
        # PiEnv expects:    [L_arm(0:7), R_arm(7:14),  L_gripper(14), R_gripper(15)]
        reordered = actions_np.copy()
        reordered[:, 7:14] = actions_np[:, 8:15]   # right arm
        reordered[:, 14]   = actions_np[:, 7]       # left gripper
        # reordered[:, 15] = actions_np[:, 15]      # right gripper (unchanged)

        return [reordered[i] for i in range(len(reordered))]   # list of (action_dim,)


def multi_image_get_item(
    raw_target, img_transform, text_tokenizer, num_image_token,
    cam_keys, dynamic_image_size, use_thumbnail,
    min_dynamic_patch, max_dynamic_patch, image_size,
):
    images, num_tiles = [], []
    for key in cam_keys:
        if key not in raw_target:
            continue
        if dynamic_image_size:
            tiles = dynamic_preprocess(
                raw_target[key], min_num=min_dynamic_patch, max_num=max_dynamic_patch,
                image_size=image_size, use_thumbnail=use_thumbnail,
            )
            images += tiles
            num_tiles.append(len(tiles))
        else:
            images.append(raw_target[key])
            num_tiles.append(1)

    pixel_values = torch.stack([img_transform(img) for img in images])
    num_patches  = pixel_values.size(0)
    num_image    = len(num_tiles)
    num_image_tokens = [num_image_token * t for t in num_tiles]

    conversation = [
        {"from": "human", "value": f"{'<image>' * num_image}{raw_target['final_prompt']}"},
        {"from": "gpt",   "value": ""},
    ]
    ret = preprocess_internvl2_5(
        "internvl2_5", [conversation], text_tokenizer,
        num_image_tokens, num_image=num_image, group_by_length=True,
    )

    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)

    img_end_id = text_tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    assert (ret["input_ids"][0] == img_end_id).sum() == num_image, \
        "Image tokens truncated — prompt may be too long"

    return dict(
        input_ids     = ret["input_ids"][0],
        attention_mask= ret["attention_mask"][0],
        position_ids  = position_ids[0],
        pixel_values  = pixel_values,
        image_flags   = torch.ones(num_patches, dtype=torch.long),
    )


# ── WebSocket server ──────────────────────────────────────────────────────────

class Go1WebsocketServer:
    def __init__(self, model: GO1InferServer, host: str = "0.0.0.0", port: int = 8999):
        self.model    = model
        self.host     = host
        self.port     = port
        self._metadata = packb({"model": "go1"})

    async def _handle(self, websocket):
        addr = websocket.remote_address
        logging.info(f"Client connected: {addr}")
        await websocket.send(self._metadata)      # protocol: metadata first
        try:
            async for message in websocket:
                obs     = unpackb(message)
                logging.info(f"[DIAG] prompt='{obs.get('prompt', '<EMPTY>')}'  state[:16]={obs['state'][:16].tolist()}")
                actions = self.model.infer(obs)   # List[np.ndarray]
                await websocket.send(packb({"actions": actions}))
        except websockets.exceptions.ConnectionClosed:
            logging.info(f"Client disconnected: {addr}")
        except Exception as e:
            logging.error(f"Inference error: {e}", exc_info=True)
            await websocket.send(str(e))

    def run(self):
        asyncio.run(self._serve())

    async def _serve(self):
        async with websockets.serve(
            self._handle, self.host, self.port,
            max_size=None, compression=None,
        ):
            logging.info(f"GO-1 WebSocket server ready → ws://{self.host}:{self.port}")
            await asyncio.Future()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GO-1 WebSocket Inference Server for Genie Sim")
    parser.add_argument("--model-path", default="/home/qkl20/workspace_lsh/AgiBot-World/experiment/genie_sim_g1/checkpoint-43685")
    parser.add_argument("--data-stats-path", default="/home/qkl20/workspace_lsh/AgiBot-World/experiment/genie_sim_g1/dataset_stats.json",
                        help="Path to dataset_stats.json; required only when model config norm=True")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8999)
    args = parser.parse_args()

    model  = GO1InferServer(args.model_path, args.data_stats_path)
    server = Go1WebsocketServer(model, host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()

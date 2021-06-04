#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, asnumpy
from habitat import Config
from habitat.core.logging import logger
from habitat.sims import make_sim
from habitat.sims.pyrobot.pyrobot import _resize_observation, PyRobotDepthSensor
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import batch_obs, generate_video
from habitat_baselines.rl.ppo import PPO
from torch import distributed

import pyrobot_utils

pyrobot_utils
from occant_baselines.config.default import get_config
from occant_baselines.models.mapnet import DepthProjectionNet
from occant_baselines.models.occant import OccupancyAnticipator
from occant_baselines.rl.ans import ActiveNeuralSLAMExplorer
from occant_baselines.rl.policy_utils import OccupancyAnticipationWrapper
from occant_baselines.supervised.imitation import Imitation
from occant_baselines.supervised.map_update import MapUpdate
from occant_utils.common import convert_gt2channel_to_gtrgb, convert_world2map
from occant_utils.visualization import generate_topdown_allocentric_map, observations_to_image


def new_get_observation(self, robot_obs, *args: Any, **kwargs: Any):
    obs = robot_obs.get(self.uuid, None)
    assert obs is not None, "Invalid observation for {} sensor".format(
        self.uuid
    )
    obs = _resize_observation(obs, self.observation_space, self.config)
    # obs = obs / MM_IN_METER  # convert from mm to m
    obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
    if self.config.NORMALIZE_DEPTH:
        # normalize depth observations to [0, 1]
        obs = (obs - self.config.MIN_DEPTH) / (
                self.config.MAX_DEPTH - self.config.MIN_DEPTH
        )
    obs = np.expand_dims(obs, axis=2)  # make depth observations a 3D array
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)

    world_size = 1
    rank = 0
    init_distributed(world_size, rank)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    locobot = LocobotExplorer(config)
    try:
        locobot._eval()
    except KeyboardInterrupt:
        sys.exit()


def init_distributed(world_size, rank):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '4301'
    distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)


class LocobotExplorer(BaseRLTrainer):

    def __init__(self, config):
        super().__init__(config)

        # Set pytorch random seed for initialization
        torch.manual_seed(config.PYT_RANDOM_SEED)

        self.mapper = None
        self.local_actor_critic = None
        self.global_actor_critic = None
        self.ans_net = None
        self.planner = None
        self.mapper_agent = None
        self.local_agent = None
        self.global_agent = None
        self.sim = None
        self.logger = logger

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.ACT_2_COMMAND = {
            0: ("go_to_relative",
                {
                    "xyt_position": [config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.forward_step, 0, 0],
                    "use_map": False,
                    "close_loop": True,
                    "smooth": False,
                },
                ),
            1: ("go_to_relative",
                {
                    "xyt_position": [0, 0, (config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.turn_angle / 180) * np.pi],
                    "use_map": False,
                    "close_loop": True,
                    "smooth": False,
                },
                ),
            2: ("go_to_relative",
                {
                    "xyt_position": [0, 0, (-config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.turn_angle / 180) * np.pi],
                    "use_map": False,
                    "close_loop": True,
                    "smooth": False,
                },
                ),
        }

        self.ACT_2_NAME = {0: 'MOVE_FORWARD',
                           1: 'TURN_LEFT',
                           2: 'TURN_RIGHT'}

    def _eval(self):

        start_time = time.time()

        if self.config.MANUAL_COMMANDS:
            init_time = None
            manual_step_start_time = None
            total_manual_time = 0.0

        checkpoint_index = int((re.findall('\d+', self.config.EVAL_CKPT_PATH_DIR))[-1])
        ckpt_dict = torch.load(self.config.EVAL_CKPT_PATH_DIR, map_location="cpu")

        print(f'Number of steps of the ckpt: {ckpt_dict["extra_state"]["step"]}')

        config = self._setup_config(ckpt_dict)
        ppo_cfg = config.RL.PPO
        ans_cfg = config.RL.ANS

        self.mapper_rollouts = None
        self._setup_actor_critic_agent(ppo_cfg, ans_cfg)

        self.mapper_agent.load_state_dict(ckpt_dict["mapper_state_dict"])
        if self.local_agent is not None:
            self.local_agent.load_state_dict(ckpt_dict["local_state_dict"])
            self.local_actor_critic = self.local_agent.actor_critic
        else:
            self.local_actor_critic = self.ans_net.local_policy
        self.global_agent.load_state_dict(ckpt_dict["global_state_dict"])
        self.mapper = self.mapper_agent.mapper
        self.global_actor_critic = self.global_agent.actor_critic

        # Set models to evaluation
        self.mapper.eval()
        self.local_actor_critic.eval()
        self.global_actor_critic.eval()

        M = ans_cfg.overall_map_size
        V = ans_cfg.MAPPER.map_size
        s = ans_cfg.MAPPER.map_scale
        imH, imW = ans_cfg.image_scale_hw

        num_steps = self.config.T_EXP

        prev_action = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        masks = torch.zeros(1, 1, device=self.device)

        try:
            self.sim = make_sim('PyRobot-v1', config=self.config.TASK_CONFIG.PYROBOT)
        except (KeyboardInterrupt, SystemExit):
            sys.exit()

        pose = defaultdict()
        self.sim._robot.camera.set_tilt(math.radians(self.config.CAMERA_TILT), wait=True)
        print(f"\nStarting Camera State: {self.sim.get_agent_state()['camera']}")
        print(f"Starting Agent State: {self.sim.get_agent_state()['base']}")
        obs = [self.sim.reset()]

        if self.config.SAVE_OBS_IMGS:
            cv2.imwrite(f'obs/depth_dirty_s.jpg', obs[0]['depth'] * 255.0)

        obs[0]['depth'][..., 0] = self._correct_depth(obs, -1)

        if self.config.SAVE_OBS_IMGS:
            cv2.imwrite(f'obs/rgb_s.jpg', obs[0]['rgb'][:, :, ::-1])
            cv2.imwrite(f'depth_s.jpg', obs[0]['depth'] * 255.0)

        starting_agent_state = self.sim.get_agent_state()
        locobot2relative = CoordProjection(starting_agent_state['base'])
        pose['base'] = locobot2relative(starting_agent_state['base'])

        print(f"Starting Agent Pose: {pose['base']}\n")
        batch = self._prepare_batch(obs, -1, device=self.device)
        if ans_cfg.MAPPER.use_sensor_positioning:
            batch['pose'] = pose['base'].to(self.device)
            batch['pose'][0][1:] = -batch['pose'][0][1:]
        prev_batch = batch

        num_envs = self.config.NUM_PROCESSES
        agent_poses_over_time = []
        for i in range(num_envs):
            agent_poses_over_time.append(torch.tensor([(M - 1) / 2, (M - 1) / 2, 0]))
        state_estimates = {
            "pose_estimates": torch.zeros(num_envs, 3).to(self.device),
            "map_states": torch.zeros(num_envs, 2, M, M).to(self.device),
            "recurrent_hidden_states": torch.zeros(
                1, num_envs, ans_cfg.LOCAL_POLICY.hidden_size
            ).to(self.device),
            "visited_states": torch.zeros(num_envs, 1, M, M).to(
                self.device
            ),
        }
        ground_truth_states = {
            "visible_occupancy": torch.zeros(num_envs, 2, M, M).to(
                self.device
            ),
            "pose": torch.zeros(num_envs, 3).to(self.device),
            "environment_layout": torch.zeros(num_envs, 2, M, M).to(self.device)
        }

        # Reset ANS states
        self.ans_net.reset()

        # Frames for video creation
        rgb_frames = []
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        step_start_time = time.time()

        for i in range(num_steps):
            print(
                f"\n\n---------------------------------------------------<<< STEP {i} >>>---------------------------------------------------")
            ep_time = torch.zeros(
                num_envs, 1, device=self.device
            ).fill_(i)

            (
                mapper_inputs,
                local_policy_inputs,
                global_policy_inputs,
                mapper_outputs,
                local_policy_outputs,
                global_policy_outputs,
                state_estimates,
                intrinsic_rewards,
            ) = self.ans_net.act(
                batch,
                prev_batch,
                state_estimates,
                ep_time,
                masks,
                deterministic=True,
            )
            if self.config.SAVE_MAP_IMGS:
                cv2.imwrite(f'maps/test_map_{i - 1}.jpg', self._round_map(state_estimates['map_states']) * 255)

            action = local_policy_outputs["actions"][0][0]

            distance2ggoal = torch.norm(
                mapper_outputs['curr_map_position'] - self.ans_net.states["curr_global_goals"], dim=1
            ) * s

            print(f"Distance to Global Goal: {distance2ggoal}")

            reached_flag = distance2ggoal < ans_cfg.goal_success_radius

            if self.config.MANUAL_COMMANDS:
                if init_time is None:
                    init_time = time.time() - start_time
                    total_manual_time = total_manual_time + init_time
                if manual_step_start_time is not None:
                    manual_step_time = time.time() - manual_step_start_time
                    total_manual_time = total_manual_time + manual_step_time
                action = torch.tensor(int(input('Waiting input to start new action: ')))
                manual_step_start_time = time.time()

                if action.item() == 3:
                    reached_flag = True

            prev_action.copy_(action)

            if not reached_flag and action.item() != 3:
                print(f'Doing Env Step [{self.ACT_2_NAME[action.item()]}]...')
                action_command = self.ACT_2_COMMAND[action.item()]

                obs = self._do_action(action_command)

                if self.config.SAVE_OBS_IMGS:
                    cv2.imwrite(f'obs/depth_dirty_{i}.jpg', obs[0]['depth'] * 255.0)

                # Correcting invalid depth pixels
                obs[0]['depth'][..., 0] = self._correct_depth(obs, i)

                if self.config.SAVE_OBS_IMGS:
                    cv2.imwrite(f'obs/rgb_{i}.jpg', obs[0]['rgb'][:, :, ::-1])
                    cv2.imwrite(f'obs/depth_{i}.jpg', obs[0]['depth'] * 255.0)

                agent_state = self.sim.get_agent_state()
                prev_batch = batch
                batch = self._prepare_batch(obs, i, device=self.device)

                pose = defaultdict()
                pose['base'] = locobot2relative(agent_state['base'])

                if ans_cfg.MAPPER.use_sensor_positioning:
                    batch['pose'] = pose['base'].to(self.device)
                    batch['pose'][0][1:] = -batch['pose'][0][1:]

                map_coords = convert_world2map(batch['pose'], (M, M),
                                               ans_cfg.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.map_scale).squeeze()
                map_coords = torch.cat((map_coords, batch['pose'][0][-1].reshape(1)))
                if self.config.COORD_DEBUG:
                    print('COORDINATES CHECK')
                    print(f'Starting Agent State: {starting_agent_state["base"]}')
                    print(f'Current Agent State: {agent_state["base"]}')
                    print(f'Current Sim Agent State: {self.sim.get_agent_state()["base"]}')
                    print(f'Current Global Coords: {batch["pose"]}')
                    print(f'Current Map Coords: {map_coords}')
                agent_poses_over_time.append(map_coords)

                step_time = time.time() - step_start_time
                print(f"\nStep Time: {step_time}")
                step_start_time = time.time()

            # Create new frame of the video
            if (
                    len(self.config.VIDEO_OPTION) > 0
            ):
                frame = observations_to_image(
                    obs[0], observation_size=300, collision_flag=self.config.DRAW_COLLISIONS
                )
                # Add ego_map_gt to frame
                ego_map_gt_i = asnumpy(batch["ego_map_gt"][0])  # (2, H, W)
                ego_map_gt_i = convert_gt2channel_to_gtrgb(ego_map_gt_i)
                ego_map_gt_i = cv2.resize(ego_map_gt_i, (300, 300))
                # frame = np.concatenate([frame], axis=1)
                # Generate ANS specific visualizations
                environment_layout = asnumpy(
                    ground_truth_states["environment_layout"][0]
                )  # (2, H, W)
                visible_occupancy = mapper_outputs["gt_mt"][0].cpu().numpy()  # (2, H, W)
                anticipated_occupancy = mapper_outputs["hat_mt"][0].cpu().numpy()  # (2, H, W)

                H = frame.shape[0]
                visible_occupancy_vis = generate_topdown_allocentric_map(
                    environment_layout,
                    visible_occupancy,
                    agent_poses_over_time,
                    thresh_explored=ans_cfg.thresh_explored,
                    thresh_obstacle=ans_cfg.thresh_obstacle,
                    zoom=False
                )
                visible_occupancy_vis = cv2.resize(
                    visible_occupancy_vis, (H, H)
                )
                anticipated_occupancy_vis = generate_topdown_allocentric_map(
                    environment_layout,
                    anticipated_occupancy,
                    agent_poses_over_time,
                    thresh_explored=ans_cfg.thresh_explored,
                    thresh_obstacle=ans_cfg.thresh_obstacle,
                    zoom=False
                )
                anticipated_occupancy_vis = cv2.resize(
                    anticipated_occupancy_vis, (H, H)
                )
                anticipated_action_map = generate_topdown_allocentric_map(
                    environment_layout,
                    anticipated_occupancy,
                    agent_poses_over_time,
                    zoom=False,
                    thresh_explored=ans_cfg.thresh_explored,
                    thresh_obstacle=ans_cfg.thresh_obstacle,
                )
                global_goals = self.ans_net.states["curr_global_goals"]
                local_goals = self.ans_net.states["curr_local_goals"]
                if global_goals is not None:
                    cX = int(global_goals[0, 0].item())
                    cY = int(global_goals[0, 1].item())
                    anticipated_action_map = cv2.circle(
                        anticipated_action_map,
                        (cX, cY),
                        10,
                        (255, 0, 0),
                        -1,
                    )
                if local_goals is not None:
                    cX = int(local_goals[0, 0].item())
                    cY = int(local_goals[0, 1].item())
                    anticipated_action_map = cv2.circle(
                        anticipated_action_map,
                        (cX, cY),
                        10,
                        (0, 255, 255),
                        -1,
                    )
                anticipated_action_map = cv2.resize(
                    anticipated_action_map, (H, H)
                )

                maps_vis = np.concatenate(
                    [
                        visible_occupancy_vis,
                        anticipated_occupancy_vis,
                        anticipated_action_map,
                        ego_map_gt_i
                    ],
                    axis=1,
                )

                if self.config.RL.ANS.overall_map_size == 2001 or self.config.RL.ANS.overall_map_size == 961:
                    if frame.shape[1] < maps_vis.shape[1]:
                        diff = maps_vis.shape[1] - frame.shape[1]
                        npad = ((0, 0), (diff // 2, diff // 2), (0, 0))
                        frame = np.pad(frame, pad_width=npad, mode='constant', constant_values=0)
                    elif frame.shape[1] > maps_vis.shape[1]:
                        diff = frame.shape[1] - maps_vis.shape[1]
                        npad = ((0, 0), (diff // 2, diff // 2), (0, 0))
                        maps_vis = np.pad(maps_vis, pad_width=npad, mode='constant', constant_values=0)
                frame = np.concatenate([frame, maps_vis], axis=0)
                rgb_frames.append(frame)
                if self.config.SAVE_VIDEO_IMGS:
                    try:
                        os.mkdir("fig1")
                    except:
                        pass
                    print("Saved imgs for Fig. 1!")
                    cv2.imwrite(f'fig1/rgb_{step_start_time}.jpg', obs[0]['rgb'][:, :, ::-1])
                    cv2.imwrite(f'fig1/depth_{step_start_time}.jpg', obs[0]['depth'] * 255.0)
                    cv2.imwrite(f'fig1/aap_{step_start_time}.jpg', anticipated_action_map[..., ::-1])
                    cv2.imwrite(f'fig1/em_{step_start_time}.jpg', ego_map_gt_i[..., ::-1])
                if self.config.DEBUG_VIDEO_FRAME:
                    cv2.imwrite('last_frame.jpg', frame)

                if reached_flag:
                    for f in range(20):
                        rgb_frames.append(frame)

                # Video creation
                video_dict = {"t": start_time}
                if (i + 1) % 10 == 0 or reached_flag:
                    generate_video(
                        video_option=self.config.VIDEO_OPTION,
                        video_dir=self.config.VIDEO_DIR,
                        images=rgb_frames,
                        episode_id=0,
                        checkpoint_idx=checkpoint_index,
                        metrics=video_dict,
                        tb_writer=TensorboardWriter('tb/locobot'),
                    )

            if reached_flag:
                if self.config.MANUAL_COMMANDS:
                    manual_step_time = time.time() - manual_step_start_time
                    total_manual_time = total_manual_time + manual_step_time
                    print(f"Manual elapsed time: {total_manual_time}")

                print(f"Number of steps: {i + 1}")
                print(f"Elapsed time: {time.time() - start_time}")
                print(f"Final Distance to Goal: {distance2ggoal}")
                if "bump" in obs[0]:
                    print(f"Collision: {obs[0]['bump']}")
                print("Exiting...")
                break
        return

    def _correct_depth(self, obs, i):
        # Inpainting, median blur and border replaced
        mask = (obs[0]['depth'] <= 0).astype(np.uint8)
        mask_dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3)), iterations=3)
        corrected_depth = (cv2.inpaint((obs[0]['depth'] * 255.0).astype(np.uint16), mask_dilated, 5,
                                       cv2.INPAINT_TELEA)).astype(np.float32) / 255.0
        median_depth = cv2.medianBlur(corrected_depth, 5)
        removed_border = median_depth[1:-1, 1:-1]
        final_depth = cv2.copyMakeBorder(removed_border, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        return final_depth

    def _setup_actor_critic_agent(self, ppo_cfg: Config, ans_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params
            ans_cfg: config node for ActiveNeuralSLAM model

        Returns:
            None
        """

        try:
            os.mkdir('video_dir')
            os.mkdir('tb')
            os.mkdir(self.config.TENSORBOARD_DIR)
        except:
            pass
        logger.add_filehandler(os.path.join(self.config.TENSORBOARD_DIR, "run.log"))

        occ_cfg = ans_cfg.OCCUPANCY_ANTICIPATOR
        mapper_cfg = ans_cfg.MAPPER
        # Create occupancy anticipation model
        occupancy_model = OccupancyAnticipator(occ_cfg)
        occupancy_model = OccupancyAnticipationWrapper(
            occupancy_model, mapper_cfg.map_size, (128, 128)
        )
        # Create ANS model
        self.ans_net = ActiveNeuralSLAMExplorer(ans_cfg, occupancy_model)
        self.mapper = self.ans_net.mapper
        self.local_actor_critic = self.ans_net.local_policy
        self.global_actor_critic = self.ans_net.global_policy
        # Create depth projection model to estimate visible occupancy
        self.depth_projection_net = DepthProjectionNet(
            ans_cfg.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION
        )
        # Set to device
        self.mapper.to(self.device)
        self.local_actor_critic.to(self.device)
        self.global_actor_critic.to(self.device)
        self.depth_projection_net.to(self.device)

        if ans_cfg.use_ddp:
            self.ans_net.to_ddp()

        # ============================== Create agents ================================
        # Mapper agent
        self.mapper_agent = MapUpdate(
            self.mapper,
            lr=mapper_cfg.lr,
            eps=mapper_cfg.eps,
            label_id=mapper_cfg.label_id,
            max_grad_norm=mapper_cfg.max_grad_norm,
            pose_loss_coef=mapper_cfg.pose_loss_coef,
            occupancy_anticipator_type=ans_cfg.OCCUPANCY_ANTICIPATOR.type,
            freeze_projection_unit=mapper_cfg.freeze_projection_unit,
            num_update_batches=mapper_cfg.num_update_batches,
            batch_size=mapper_cfg.map_batch_size,
            mapper_rollouts=self.mapper_rollouts,
        )
        # Local policy
        if ans_cfg.LOCAL_POLICY.use_heuristic_policy:
            self.local_agent = None
        elif ans_cfg.LOCAL_POLICY.learning_algorithm == "rl":
            self.local_agent = PPO(
                actor_critic=self.local_actor_critic,
                clip_param=ppo_cfg.clip_param,
                ppo_epoch=ppo_cfg.ppo_epoch,
                num_mini_batch=ppo_cfg.num_mini_batch,
                value_loss_coef=ppo_cfg.value_loss_coef,
                entropy_coef=ppo_cfg.local_entropy_coef,
                lr=ppo_cfg.local_policy_lr,
                eps=ppo_cfg.eps,
                max_grad_norm=ppo_cfg.max_grad_norm,
            )
        else:
            self.local_agent = Imitation(
                actor_critic=self.local_actor_critic,
                lr=ppo_cfg.local_policy_lr,
                eps=ppo_cfg.eps,
                max_grad_norm=ppo_cfg.max_grad_norm,
            )
        # Global policy
        self.global_agent = PPO(
            actor_critic=self.global_actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )
        if ans_cfg.model_path != "":
            self.resume_checkpoint(ans_cfg.model_path)

    def _setup_config(self, ckpt_dict):
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()
        if "COLLISION_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
            config.TASK_CONFIG.TASK.SENSORS.append("COLLISION_SENSOR")
        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_EXP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
        self.logger.info(f"env config: {config}")
        return config

    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()
        config.defrost()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)
        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = "val"

        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

    def _prepare_batch(self, observations, i, device=None, actions=None):
        imH, imW = self.config.RL.ANS.image_scale_hw
        device = self.device if device is None else device

        batch = batch_obs(observations, device=device)

        if batch["rgb"].size(1) != imH or batch["rgb"].size(2) != imW:
            rgb = rearrange(batch["rgb"], "b h w c -> b c h w")
            rgb = F.interpolate(rgb, (imH, imW), mode="bilinear")
            batch["rgb"] = rearrange(rgb, "b c h w -> b h w c")
        if batch["depth"].size(1) != imH or batch["depth"].size(2) != imW:
            depth = rearrange(batch["depth"], "b h w c -> b c h w")
            depth = F.interpolate(depth, (imH, imW), mode="bilinear")
            batch["depth"] = rearrange(depth, "b c h w -> b h w c")

        # Compute ego_map_gt from depth
        ego_map_gt_b = self.depth_projection_net(
            rearrange(batch["depth"], "b h w c -> b c h w")
        )
        batch["ego_map_gt"] = rearrange(ego_map_gt_b, "b c h w -> b h w c")

        if actions is None:
            batch["prev_actions"] = torch.zeros(1, 1).to(self.device)
        else:
            batch["prev_actions"] = actions

        return batch

    def _do_action(self, action_command):
        obs = [self.sim.step(action_command[0], action_command[1])]
        return obs

    def _round_map(self, sem_map):
        new_map = sem_map.cpu().numpy()[0, 0]
        new_map[new_map >= 0.5] = 1.0
        new_map[new_map < 0.5] = 0.0
        return new_map


class CoordProjection:
    def __init__(self, starting_pose, debug=False):
        self.delta_x = starting_pose[0]
        self.delta_y = starting_pose[1]
        self.yaw = starting_pose[-1]
        self.rot_matrix = torch.tensor(
            [
                [np.cos(self.yaw), -np.sin(self.yaw), self.delta_x],
                [np.sin(self.yaw), np.cos(self.yaw), self.delta_y],
                [0, 0, 1]
            ],
            dtype=torch.float32)
        self.debug = debug

    def _coordinate_projection(self, current_pose):
        agent_state = torch.tensor([current_pose[0], current_pose[1], 1], dtype=torch.float32).unsqueeze(1)
        agent_state = torch.mm(torch.inverse(self.rot_matrix), agent_state)
        agent_state[-1] = current_pose[-1] - self.yaw

        if self.debug:
            print(f'Rotation Matrix: {self.rot_matrix}')
            print(f'Agent State: {agent_state}')

        return agent_state.reshape((1, 3))

    def __call__(self, *args, **kwargs):
        return self._coordinate_projection(*args, **kwargs)


if __name__ == "__main__":
    PyRobotDepthSensor.get_observation = new_get_observation

    try:
        main()
    except KeyboardInterrupt or EOFError:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

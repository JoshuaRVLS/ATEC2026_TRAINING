# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../parkour_tasks")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../source/atec_rl_lab")))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--debug_play", action="store_true", default=False, help="Print and save play diagnostics.")
parser.add_argument("--debug_play_interval", type=int, default=25, help="Step interval for play diagnostics.")
parser.add_argument("--debug_play_env", type=int, default=0, help="Environment index used for play diagnostics.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*args, **kwargs):
        return None
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg

from scripts.rsl_rl.exporter import (
export_teacher_policy_as_jit, 
export_teacher_policy_as_onnx,
export_deploy_policy_as_jit, 
export_deploy_policy_as_onnx,
)
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

import parkour_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def _get_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "get"):
        policy_value = value.get("policy", None)
        if policy_value is not None:
            return policy_value
    return None


def _format_stats(name, value, env_id=None):
    tensor = _get_tensor(value)
    if tensor is None:
        return f"{name}=n/a"
    if env_id is not None and tensor.ndim > 1:
        tensor = tensor[env_id]
    tensor = tensor.detach().float()
    if tensor.numel() == 0:
        return f"{name}=empty"
    return (
        f"{name}[mean={tensor.mean().item():+.4f}, std={tensor.std(unbiased=False).item():+.4f}, "
        f"min={tensor.min().item():+.4f}, max={tensor.max().item():+.4f}]"
    )


def _safe_action_tensor(action_term, attr_name):
    value = getattr(action_term, attr_name, None)
    if value is None:
        value = getattr(action_term, f"_{attr_name}", None)
    return value


def _safe_scalar(value, default=0.0):
    if isinstance(value, torch.Tensor):
        return value.detach().float().item()
    if value is None:
        return default
    return float(value)


def _build_play_debugger(env, log_dir, env_id):
    env_id = max(0, min(env_id, env.unwrapped.num_envs - 1))
    csv_path = os.path.join(log_dir, "play_debug.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "step",
            "episode_len",
            "reward",
            "done",
            "cmd_x",
            "vel_x",
            "base_z",
            "roll",
            "pitch",
            "action_abs_mean",
            "processed_action_std",
            "contact_count",
            "target_dist",
            "goal_idx",
            "terrain_level",
            "obs_min",
            "obs_max",
            "lidar_min",
            "lidar_max",
        ]
    )
    print(f"[DEBUG_PLAY] writing diagnostics to: {csv_path}")

    def log(step, obs, actions, rew, dones):
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        action_term = unwrapped.action_manager.get_term("joint_pos")
        command = unwrapped.command_manager.get_command("base_velocity")

        root_quat_w = robot.data.root_quat_w
        roll, pitch, yaw = euler_xyz_from_quat(root_quat_w)
        roll = wrap_to_pi(roll)
        pitch = wrap_to_pi(pitch)
        yaw = wrap_to_pi(yaw)

        raw_actions = _safe_action_tensor(action_term, "raw_actions")
        processed_actions = _safe_action_tensor(action_term, "processed_actions")
        joint_ids = getattr(action_term, "joint_ids", slice(None))
        joint_pos_rel = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
        joint_vel = robot.data.joint_vel[:, joint_ids]

        contact_count = -1
        contact_text = "contacts=n/a"
        contact_sensor = unwrapped.scene.sensors.get("contact_forces", None)
        if contact_sensor is not None:
            try:
                foot_ids = contact_sensor.find_bodies(".*_foot")[0]
                forces = contact_sensor.data.net_forces_w_history[env_id, 0, foot_ids]
                contacts = torch.norm(forces, dim=-1) > 2.0
                contact_count = int(contacts.sum().item())
                contact_text = f"contacts={contacts.detach().cpu().int().tolist()} count={contact_count}"
            except Exception as exc:
                contact_text = f"contacts=error({type(exc).__name__})"

        target_dist = 0.0
        goal_idx = 0.0
        terrain_level = 0.0
        parkour_text = "parkour=n/a"
        try:
            parkour_event = unwrapped.parkour_manager.get_term("base_parkour")
            target_rel = parkour_event.target_pos_rel[env_id]
            target_dist = torch.norm(target_rel).detach().float().item()
            goal_idx = parkour_event.cur_goal_idx[env_id].detach().float().item()
            terrain_level = parkour_event.terrain.terrain_levels[env_id].detach().float().item()
            parkour_text = (
                f"target_rel={target_rel.detach().cpu().tolist()} "
                f"target_dist={target_dist:.3f} goal_idx={goal_idx:.0f} terrain_level={terrain_level:.0f}"
            )
        except Exception as exc:
            parkour_text = f"parkour=error({type(exc).__name__})"

        lidar_min = 0.0
        lidar_max = 0.0
        lidar_text = "lidar=n/a"
        ray_sensor = unwrapped.scene.sensors.get("height_scanner", None)
        if ray_sensor is not None:
            try:
                heights = ray_sensor.data.pos_w[env_id, 2] - ray_sensor.data.ray_hits_w[env_id, ..., 2] - 0.3
                heights = torch.clip(heights.detach().float(), -1.0, 1.0)
                lidar_min = heights.min().item()
                lidar_max = heights.max().item()
                lidar_clip_low = int((heights <= -0.999).sum().item())
                lidar_clip_high = int((heights >= 0.999).sum().item())
                lidar_text = (
                    f"lidar[mean={heights.mean().item():+.3f}, min={lidar_min:+.3f}, max={lidar_max:+.3f}, "
                    f"clip_low={lidar_clip_low}, clip_high={lidar_clip_high}]"
                )
            except Exception as exc:
                lidar_text = f"lidar=error({type(exc).__name__})"

        obs_tensor = _get_tensor(obs)
        obs_min = obs_tensor[env_id].detach().float().min().item() if obs_tensor is not None and obs_tensor.ndim > 1 else 0.0
        obs_max = obs_tensor[env_id].detach().float().max().item() if obs_tensor is not None and obs_tensor.ndim > 1 else 0.0
        obs_clip_text = "obs_clip=n/a"
        if obs_tensor is not None and obs_tensor.ndim > 1:
            obs_env = obs_tensor[env_id].detach().float()
            obs_clip_low = int((obs_env <= -99.9).sum().item())
            obs_clip_high = int((obs_env >= 99.9).sum().item())
            obs_clip_text = f"obs_clip_low={obs_clip_low} obs_clip_high={obs_clip_high}"

        action_env = actions[env_id].detach().float()
        processed_env = processed_actions[env_id].detach().float() if processed_actions is not None else action_env
        reward_value = _safe_scalar(rew[env_id] if isinstance(rew, torch.Tensor) and rew.ndim > 0 else rew)
        done_value = int(_safe_scalar(dones[env_id] if isinstance(dones, torch.Tensor) and dones.ndim > 0 else dones))
        episode_len = int(unwrapped.episode_length_buf[env_id].detach().cpu().item())
        cmd_x = command[env_id, 0].detach().float().item()
        vel_x = robot.data.root_lin_vel_b[env_id, 0].detach().float().item()
        base_z = robot.data.root_pos_w[env_id, 2].detach().float().item()

        print(
            f"[DEBUG_PLAY step={step} env={env_id}] ep_len={episode_len} rew={reward_value:+.4f} done={done_value} "
            f"cmd={command[env_id].detach().cpu().tolist()} "
            f"vel_b={robot.data.root_lin_vel_b[env_id].detach().cpu().tolist()} "
            f"base_z={base_z:+.3f} rpy=({roll[env_id].item():+.3f},{pitch[env_id].item():+.3f},{yaw[env_id].item():+.3f})"
        )
        print(
            "[DEBUG_PLAY] "
            + " | ".join(
                [
                    _format_stats("obs", obs, env_id),
                    obs_clip_text,
                    _format_stats("policy_action", actions, env_id),
                    _format_stats("raw_action", raw_actions, env_id),
                    _format_stats("processed_action", processed_actions, env_id),
                    _format_stats("joint_pos_rel", joint_pos_rel, env_id),
                    _format_stats("joint_vel", joint_vel, env_id),
                ]
            )
        )
        print(f"[DEBUG_PLAY] {contact_text} | {parkour_text} | {lidar_text}")

        csv_writer.writerow(
            [
                step,
                episode_len,
                reward_value,
                done_value,
                cmd_x,
                vel_x,
                base_z,
                roll[env_id].detach().float().item(),
                pitch[env_id].detach().float().item(),
                torch.mean(torch.abs(action_env)).item(),
                processed_env.std(unbiased=False).item(),
                contact_count,
                target_dist,
                goal_idx,
                terrain_level,
                obs_min,
                obs_max,
                lidar_min,
                lidar_max,
            ]
        )
        csv_file.flush()

    return log, csv_file


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.algorithm.class_name in ["PPOWithExtractor", "DistillationWithExtractor"]:
        ppo_runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        env.return_tensordict = True
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(ppo_runner)
    # obtain the trained policy for inference

    if agent_cfg.algorithm.class_name not in ["PPOWithExtractor", "DistillationWithExtractor"]:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    elif agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
        policy = ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        policy_nn = ppo_runner.alg.depth_actor
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_deploy")
        export_deploy_policy_as_jit(policy_nn, 
                                    estimator,
                                    depth_encoder,
                                    ppo_runner.obs_normalizer, 
                                    path=export_model_dir, 
                                    filename="policy.pt")
        export_deploy_policy_as_onnx(
                            policy_nn, 
                            estimator,
                            depth_encoder,
                            agent_cfg,
                            normalizer=ppo_runner.obs_normalizer, 
                            path=export_model_dir, 
                            filename="policy.onnx"
                        )

    else:
        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = ppo_runner.alg.policy
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_teacher")
        export_teacher_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_teacher_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    dt = env.unwrapped.step_dt
    if agent_cfg.algorithm.class_name in ["PPOWithExtractor", "DistillationWithExtractor"]:
        estimator_paras = agent_cfg.to_dict()["estimator"]
        num_prop = estimator_paras["num_prop"]
        num_scan = estimator_paras["num_scan"]
        num_priv_explicit = estimator_paras["num_priv_explicit"]
    # reset environment
    observations = env.get_observations()
    if isinstance(observations, tuple):
        obs, extras = observations
    else:
        obs, extras = observations, {"observations": observations}
    timestep = 0
    debug_log = None
    debug_csv_file = None
    if args_cli.debug_play:
        debug_log, debug_csv_file = _build_play_debugger(env, log_dir, args_cli.debug_play_env)
    # simulate environment
    try:
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            if agent_cfg.algorithm.class_name not in ["PPOWithExtractor", "DistillationWithExtractor"]:
                with torch.inference_mode():
                    actions = policy(obs)
            elif agent_cfg.algorithm.class_name != "DistillationWithExtractor":
                with torch.inference_mode():
                    # agent stepping
                    obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                    actions = policy(obs, hist_encoding = True)
                # env stepping
            else:
                depth_camera = extras["observations"]['depth_camera'].to(env.device)
                with torch.inference_mode():
                    if env.unwrapped.common_step_counter %5 == 0:
                        obs_student = obs[:, :num_prop].clone()
                        obs_student[:, 6:8] = 0
                        depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
                        depth_latent = depth_latent_and_yaw[:, :-2]
                        yaw = depth_latent_and_yaw[:, -2:]
                    obs[:, 6:8] = 1.5*yaw
                    # obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                    actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
            obs, rew, dones, extras = env.step(actions)
            if debug_log is not None and timestep % max(args_cli.debug_play_interval, 1) == 0:
                debug_log(timestep, obs, actions, rew, dones)
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            else:
                timestep += 1

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if debug_csv_file is not None:
            debug_csv_file.close()

    # # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

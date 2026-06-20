from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi, quat_apply
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from parkour_isaaclab.managers.parkour_manager import sanitize_env_ids
from collections.abc import Sequence

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import cv2
import numpy as np 

def _joint_pos_action_term(env: ParkourManagerBasedRLEnv):
    return env.action_manager.get_term("joint_pos")

def _controlled_joint_ids(env: ParkourManagerBasedRLEnv):
    return getattr(_joint_pos_action_term(env), "joint_ids", slice(None))

def _controlled_action_dim(env: ParkourManagerBasedRLEnv) -> int:
    return _joint_pos_action_term(env).raw_actions.shape[-1]

class reward_feet_edge(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.parkour_event: ParkourEvent =  env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.base_body_name = cfg.params.get("base_body_name", "base")
        self.body_id = self.contact_sensor.find_bodies(self.base_body_name)[0]
        self.horizontal_scale = env.scene.terrain.cfg.terrain_generator.horizontal_scale
        size_x, size_y = env.scene.terrain.cfg.terrain_generator.size
        self.rows_offset = (size_x * env.scene.terrain.cfg.terrain_generator.num_rows/2)
        self.cols_offset = (size_y * env.scene.terrain.cfg.terrain_generator.num_cols/2)
        total_x_edge_maskes = torch.from_numpy(self.parkour_event.terrain.terrain_generator_class.x_edge_maskes).to(device = self.device)
        self.x_edge_masks_tensor = total_x_edge_maskes.permute(0, 2, 1, 3).reshape(
            env.scene.terrain.terrain_generator_class.total_width_pixels, env.scene.terrain.terrain_generator_class.total_length_pixels
        )

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        base_body_name: str = "base",
        ) -> torch.Tensor:
        feet_pos_x = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,0] + self.rows_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_y = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,1] + self.cols_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_x = torch.clip(feet_pos_x, 0, self.x_edge_masks_tensor.shape[0]-1)
        feet_pos_y = torch.clip(feet_pos_y, 0, self.x_edge_masks_tensor.shape[1]-1)
        feet_at_edge = self.x_edge_masks_tensor[feet_pos_x, feet_pos_y]
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        contact = torch.norm(contact_forces, dim=-1) > 2.
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        self.feet_at_edge = contact_filt & feet_at_edge
        rew = (self.parkour_event.terrain.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        ## This is for debugging to matching index and x_edge_mask
        # origin = self.x_edge_masks_tensor.detach().cpu().numpy().astype(np.uint8) * 255
        # cv2.imshow('origin',origin)
        # origin[feet_pos_x.detach().cpu().numpy(), feet_pos_y.detach().cpu().numpy()] -= 100
        # cv2.imshow('feet_edge',origin)
        # cv2.waitKey(1)
        return rew

def reward_torques(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, _controlled_joint_ids(env)]), dim=1)

def reward_dof_error(    
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _controlled_joint_ids(env)
    return torch.sum(torch.square(asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]), dim=1)

def reward_hip_pos(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] \
                                    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)

def reward_ang_vel_xy(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:,:2]), dim=1)

class reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        action_dim = _controlled_action_dim(env)
        self.previous_actions = torch.zeros(env.num_envs, 2, action_dim, dtype= torch.float ,device=self.device)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = sanitize_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() > 0:
            self.previous_actions[env_ids] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        self.previous_actions[:, 1, :] = env.action_manager.get_term('joint_pos').raw_actions
        return torch.norm(self.previous_actions[:, 1, :] - self.previous_actions[:,0,:], dim=1)
    
class reward_dof_acc(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.joint_ids = _controlled_joint_ids(env)
        action_dim = _controlled_action_dim(env)
        self.previous_joint_vel = torch.zeros(env.num_envs, 2, action_dim, dtype= torch.float ,device=self.device)
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = sanitize_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() > 0:
            self.previous_joint_vel[env_ids] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_joint_vel[:, 0, :] = self.previous_joint_vel[:, 1, :]
        self.previous_joint_vel[:, 1, :] = asset.data.joint_vel[:, self.joint_ids]
        return torch.sum(torch.square((self.previous_joint_vel[:, 1, :] - self.previous_joint_vel[:,0,:]) / self.dt), dim=1)
        
def reward_lin_vel_z(
    env: ParkourManagerBasedRLEnv,        
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.square(asset.data.root_lin_vel_b[:, 2])
    rew[(terrain_names !='parkour_flat')[:,-1]] *= 0.5
    return rew

def reward_orientation(
    env: ParkourManagerBasedRLEnv,   
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    rew[(terrain_names !='parkour_flat')[:,-1]] = 0.
    return rew

def reward_feet_stumble(
    env: ParkourManagerBasedRLEnv,        
    sensor_cfg: SceneEntityCfg ,
    ) -> torch.Tensor: 
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    rew = torch.any(torch.norm(net_contact_forces[:, :, :2], dim=2) >\
            4 *torch.abs(net_contact_forces[:, :, 2]), dim=1)
    return rew.float()

def reward_tracking_goal_vel(
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    target_pos_rel = parkour_event.target_pos_rel
    target_vel = target_pos_rel / (torch.norm(target_pos_rel, dim=-1, keepdim=True) + 1e-5)
    cur_vel = asset.data.root_vel_w[:, :2]
    proj_vel = torch.sum(target_vel * cur_vel, dim=-1)
    command_vel = env.command_manager.get_command('base_velocity')[:, 0]
    rew_move = torch.minimum(proj_vel, command_vel) / (command_vel + 1e-5)
    return rew_move

def reward_tracking_goal_vel_positive(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    target_pos_rel = parkour_event.target_pos_rel
    target_vel = target_pos_rel / (torch.norm(target_pos_rel, dim=-1, keepdim=True) + 1e-5)
    proj_vel = torch.sum(target_vel * asset.data.root_vel_w[:, :2], dim=-1)
    command_vel = env.command_manager.get_command("base_velocity")[:, 0]
    return torch.clamp(proj_vel / (command_vel + 1e-5), min=0.0, max=1.5)

def reward_track_forward_velocity(
    env: ParkourManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.25,
    max_roll: float = 0.75,
    max_pitch: float = 0.75,
    min_height: float = 0.15,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.05)
    forward_vel = asset.data.root_lin_vel_b[:, 0]
    reward = torch.exp(-torch.square(command_x - forward_vel) / (std * std))
    return reward * _upright_mask(env, asset_cfg, max_roll, max_pitch, min_height).float()

def reward_track_lin_vel_xy_exp(
    env: ParkourManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy = torch.zeros_like(asset.data.root_lin_vel_b[:, :2])
    command_xy[:, 0] = command[:, 0]
    if command.shape[1] > 1:
        command_xy[:, 1] = command[:, 1]
    velocity_error = torch.sum(torch.square(command_xy - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-velocity_error / (std * std))

def reward_forward_velocity_positive(
    env: ParkourManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_ratio: float = 1.5,
    max_roll: float = 0.75,
    max_pitch: float = 0.75,
    min_height: float = 0.15,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.05)
    forward_vel = asset.data.root_lin_vel_b[:, 0]
    reward = torch.clamp(forward_vel / command_x, min=0.0, max=max_ratio)
    return reward * _upright_mask(env, asset_cfg, max_roll, max_pitch, min_height).float()

def reward_backward_velocity(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(torch.clamp(-asset.data.root_lin_vel_b[:, 0], min=0.0))

def reward_base_height(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.53,
    max_error: float = 0.5,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.clamp(asset.data.root_pos_w[:, 2] - target_height, min=-max_error, max=max_error)
    return torch.square(height_error)

class reward_forward_displacement(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.parkour_event: ParkourEvent = env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.previous_pos = self.asset.data.root_pos_w[:, :2].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = sanitize_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() > 0:
            self.previous_pos[env_ids] = self.asset.data.root_pos_w[env_ids, :2]

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        parkour_name: str,
        command_name: str = "base_velocity",
        clip: float = 0.08,
    ) -> torch.Tensor:
        current_pos = self.asset.data.root_pos_w[:, :2]
        target_direction = self.parkour_event.target_pos_rel
        target_direction = target_direction / (torch.norm(target_direction, dim=-1, keepdim=True) + 1e-5)
        displacement = torch.sum((current_pos - self.previous_pos) * target_direction, dim=-1)
        displacement = torch.clamp(displacement, min=-clip, max=clip)
        self.previous_pos[:] = current_pos
        command_x = env.command_manager.get_command(command_name)[:, 0]
        reward = torch.clamp(displacement / env.step_dt, min=0.0)
        reward *= command_x > 0.1
        return reward * _upright_mask(env, asset_cfg).float()

class reward_no_forward_progress(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.parkour_event: ParkourEvent = env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.previous_pos = self.asset.data.root_pos_w[:, :2].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = sanitize_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() > 0:
            self.previous_pos[env_ids] = self.asset.data.root_pos_w[env_ids, :2]

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        parkour_name: str,
        command_name: str = "base_velocity",
        min_speed: float = 0.05,
    ) -> torch.Tensor:
        current_pos = self.asset.data.root_pos_w[:, :2]
        target_direction = self.parkour_event.target_pos_rel
        target_direction = target_direction / (torch.norm(target_direction, dim=-1, keepdim=True) + 1e-5)
        forward_speed = torch.sum((current_pos - self.previous_pos) * target_direction, dim=-1) / env.step_dt
        self.previous_pos[:] = current_pos
        command_x = env.command_manager.get_command(command_name)[:, 0]
        penalty = (command_x > 0.1) & (forward_speed < min_speed)
        return penalty.float() * _upright_mask(env, asset_cfg).float()

def reward_feet_air_time(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum(torch.clamp(last_air_time - threshold, min=0.0) * first_contact, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward * _upright_mask(env).float()

def reward_feet_contact_count(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    expect_contact_num: int,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    penalty = (contact_num != expect_contact_num).float()
    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return penalty * _upright_mask(env).float()

class reward_trot_gait(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        self.synced_feet_pairs = [
            self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0],
            self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0],
        ]

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        synced_feet_pair_names,
        std: float = 0.5,
        max_err: float = 0.2,
        command_threshold: float = 0.1,
    ) -> torch.Tensor:
        sync_reward = self._sync_reward(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1], std, max_err)
        sync_reward *= self._sync_reward(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1], std, max_err)
        async_reward = self._async_reward(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0], std, max_err)
        async_reward *= self._async_reward(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1], std, max_err)
        async_reward *= self._async_reward(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1], std, max_err)
        async_reward *= self._async_reward(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1], std, max_err)
        reward = sync_reward * async_reward
        reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
        return reward * _upright_mask(env).float()

    def _sync_reward(self, foot_0: int, foot_1: int, std: float, max_err: float) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_air = torch.clamp(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=max_err * max_err)
        se_contact = torch.clamp(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=max_err * max_err)
        return torch.exp(-(se_air + se_contact) / std)

    def _async_reward(self, foot_0: int, foot_1: int, std: float, max_err: float) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_act_0 = torch.clamp(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=max_err * max_err)
        se_act_1 = torch.clamp(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=max_err * max_err)
        return torch.exp(-(se_act_0 + se_act_1) / std)

def reward_joint_mirror(
    env: ParkourManagerBasedRLEnv,
    mirror_joints: list[list[str]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "parkour_joint_mirror_cache"):
        env.parkour_joint_mirror_cache = [
            [asset.find_joints(joint_name)[0] for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.parkour_joint_mirror_cache:
        reward += torch.sum(torch.square(asset.data.joint_pos[:, joint_pair[0]] - asset.data.joint_pos[:, joint_pair[1]]), dim=-1)
    reward *= 1.0 / max(len(mirror_joints), 1)
    return reward * _upright_mask(env, asset_cfg).float()

def reward_action_mirror(
    env: ParkourManagerBasedRLEnv,
    mirror_joints: list[list[str]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "parkour_action_mirror_cache"):
        action_joint_ids = _controlled_joint_ids(env)
        action_joint_ids = action_joint_ids.tolist() if hasattr(action_joint_ids, "tolist") else list(action_joint_ids)
        action_index_by_joint_id = {int(joint_id): index for index, joint_id in enumerate(action_joint_ids)}
        env.parkour_action_mirror_cache = []
        for joint_pair in mirror_joints:
            left_ids = asset.find_joints(joint_pair[0])[0]
            right_ids = asset.find_joints(joint_pair[1])[0]
            left_action_ids = [action_index_by_joint_id[int(joint_id)] for joint_id in left_ids]
            right_action_ids = [action_index_by_joint_id[int(joint_id)] for joint_id in right_ids]
            env.parkour_action_mirror_cache.append((left_action_ids, right_action_ids))
    actions = env.action_manager.get_term("joint_pos").raw_actions
    reward = torch.zeros(env.num_envs, device=env.device)
    for left_action_ids, right_action_ids in env.parkour_action_mirror_cache:
        left_actions = torch.abs(actions[:, left_action_ids])
        right_actions = torch.abs(actions[:, right_action_ids])
        reward += torch.sum(torch.square(left_actions - right_actions), dim=-1)
    reward *= 1.0 / max(len(mirror_joints), 1)
    return reward * _upright_mask(env, asset_cfg).float()

def reward_feet_height_body(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_rel = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    foot_vel_rel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    foot_pos_b = torch.zeros_like(foot_pos_rel)
    foot_vel_b = torch.zeros_like(foot_vel_rel)
    for foot_id in range(len(asset_cfg.body_ids)):
        foot_pos_b[:, foot_id, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, foot_pos_rel[:, foot_id, :])
        foot_vel_b[:, foot_id, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, foot_vel_rel[:, foot_id, :])
    foot_z_error = torch.square(foot_pos_b[:, :, 2] - target_height)
    foot_speed = torch.tanh(tanh_mult * torch.norm(foot_vel_b[:, :, :2], dim=2))
    reward = torch.sum(foot_z_error * foot_speed, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward * _upright_mask(env, SceneEntityCfg("robot")).float()

def reward_feet_slide(
    env: ParkourManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_rel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    foot_vel_b = torch.zeros_like(foot_vel_rel)
    for foot_id in range(len(asset_cfg.body_ids)):
        foot_vel_b[:, foot_id, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, foot_vel_rel[:, foot_id, :])
    foot_lateral_vel = torch.sqrt(torch.sum(torch.square(foot_vel_b[:, :, :2]), dim=2))
    return torch.sum(foot_lateral_vel * contacts, dim=1) * _upright_mask(env, SceneEntityCfg("robot")).float()

def reward_feet_contact_without_cmd(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    return reward * _upright_mask(env).float()

def reward_upward(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(1.0 - asset.data.projected_gravity_b[:, 2])

def _upright_mask(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_roll: float = 0.75,
    max_pitch: float = 0.75,
    min_height: float = 0.15,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return (
        (torch.abs(wrap_to_pi(roll)) < max_roll)
        & (torch.abs(wrap_to_pi(pitch)) < max_pitch)
        & (asset.data.root_pos_w[:, 2] > min_height)
    )

def reward_upright_alive(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_roll: float = 0.75,
    max_pitch: float = 0.75,
    min_height: float = 0.15,
) -> torch.Tensor:
    return _upright_mask(env, asset_cfg, max_roll, max_pitch, min_height).float()

def reward_fall_penalty(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_roll: float = 1.0,
    max_pitch: float = 1.0,
    min_height: float = 0.0,
) -> torch.Tensor:
    return 1.0 - _upright_mask(env, asset_cfg, max_roll, max_pitch, min_height).float()

class reward_progress_to_goal(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.parkour_event: ParkourEvent = env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_distance = torch.zeros(env.num_envs, device=self.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = sanitize_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() == 0:
            return
        robot_root_pos_w = self.asset.data.root_pos_w[env_ids, :2] - self.parkour_event.env_origins[env_ids, :2]
        self.previous_distance[env_ids] = torch.norm(
            self.parkour_event.cur_goals[env_ids, :2] - robot_root_pos_w, dim=-1
        )

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        parkour_name: str,
        asset_cfg: SceneEntityCfg,
        clip: float = 0.25,
    ) -> torch.Tensor:
        robot_root_pos_w = self.asset.data.root_pos_w[:, :2] - self.parkour_event.env_origins[:, :2]
        current_distance = torch.norm(self.parkour_event.cur_goals[:, :2] - robot_root_pos_w, dim=-1)
        progress = torch.clamp(self.previous_distance - current_distance, min=-clip, max=clip)
        self.previous_distance[:] = current_distance
        return (progress / env.step_dt) * _upright_mask(env, asset_cfg).float()

class reward_progress_from_start(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.parkour_event: ParkourEvent = env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.previous_distance = torch.zeros(env.num_envs, device=self.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = sanitize_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() > 0:
            self.previous_distance[env_ids] = self.parkour_event.dis_to_start_pos[env_ids]

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        parkour_name: str,
        clip: float = 0.25,
    ) -> torch.Tensor:
        current_distance = self.parkour_event.dis_to_start_pos
        progress = torch.clamp(current_distance - self.previous_distance, min=-clip, max=clip)
        self.previous_distance[:] = current_distance
        return (progress / env.step_dt) * _upright_mask(env).float()

def reward_goal_reached(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str,
) -> torch.Tensor:
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    if not hasattr(parkour_event, "reached_goal_ids"):
        return torch.zeros(env.num_envs, device=env.device)
    return parkour_event.reached_goal_ids.float()

def reward_tracking_yaw(     
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    yaw = torch.atan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
                    1 - 2*(q[:,2]**2 + q[:,3]**2))
    return torch.exp(-torch.abs((parkour_event.target_yaw - yaw)))

def reward_yaw_when_moving(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    target_pos_rel = parkour_event.target_pos_rel
    target_vel = target_pos_rel / (torch.norm(target_pos_rel, dim=-1, keepdim=True) + 1e-5)
    proj_vel = torch.sum(target_vel * asset.data.root_vel_w[:, :2], dim=-1)
    moving_scale = torch.clamp(proj_vel / 0.3, min=0.0, max=1.0)
    return moving_scale * torch.exp(-torch.abs(parkour_event.target_yaw - yaw))

class reward_delta_torques(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.joint_ids = _controlled_joint_ids(env)
        action_dim = _controlled_action_dim(env)
        self.previous_torque = torch.zeros(env.num_envs, 2, action_dim, dtype= torch.float ,device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = sanitize_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() > 0:
            self.previous_torque[env_ids] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_torque[:, 0, :] = self.previous_torque[:, 1, :]
        self.previous_torque[:, 1, :] = self.asset.data.applied_torque[:, self.joint_ids]
        return torch.sum(torch.square((self.previous_torque[:, 1, :] - self.previous_torque[:,0,:])), dim=1)

def reward_collision(
    env: ParkourManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg ,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    return torch.sum(1.*(torch.norm(net_contact_forces, dim=-1) > 0.1), dim=1)

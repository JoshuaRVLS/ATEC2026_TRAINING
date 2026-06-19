from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ParkourEventsCfg,
    TeacherObservationsCfg,
    TeacherRewardsCfg,
    TerminationsCfg,
)
from parkour_isaaclab.envs.mdp import observations, rewards, terminations


B2_BASE_BODY = "base_link"
B2_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
B2_LEG_JOINTS = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


@configclass
class B2TeacherObservationsCfg(TeacherObservationsCfg):
    def __post_init__(self):
        self.policy.extreme_parkour_observations.params["base_body_name"] = B2_BASE_BODY


@configclass
class B2ProprioLidarObservationsCfg(TeacherObservationsCfg):
    def __post_init__(self):
        self.policy.extreme_parkour_observations.func = observations.ProprioLidarParkourObservations
        self.policy.extreme_parkour_observations.params["base_body_name"] = B2_BASE_BODY


@configclass
class B2TeacherRewardsCfg(TeacherRewardsCfg):
    def __post_init__(self):
        self.reward_collision.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[B2_BASE_BODY, ".*_calf", ".*_thigh"]
        )
        self.reward_feet_edge.params["asset_cfg"] = SceneEntityCfg(name="robot", body_names=B2_FEET)
        self.reward_feet_edge.params["base_body_name"] = B2_BASE_BODY


@configclass
class B2ProprioLidarRewardsV2Cfg(B2TeacherRewardsCfg):
    reward_forward_displacement = RewTerm(
        func=rewards.reward_forward_displacement,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "clip": 0.08,
        },
    )
    reward_no_forward_progress = RewTerm(
        func=rewards.reward_no_forward_progress,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "min_speed": 0.05,
        },
    )
    reward_track_forward_velocity = RewTerm(
        func=rewards.reward_track_forward_velocity,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "std": 0.25,
            "max_roll": 0.75,
            "max_pitch": 0.75,
            "min_height": 0.15,
        },
    )
    reward_forward_velocity_positive = RewTerm(
        func=rewards.reward_forward_velocity_positive,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "max_ratio": 1.5,
            "max_roll": 0.75,
            "max_pitch": 0.75,
            "min_height": 0.15,
        },
    )
    reward_backward_velocity = RewTerm(
        func=rewards.reward_backward_velocity,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_joint_mirror = RewTerm(
        func=rewards.reward_joint_mirror,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ],
        },
    )
    reward_action_mirror = RewTerm(
        func=rewards.reward_action_mirror,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ],
        },
    )
    reward_feet_height_body = RewTerm(
        func=rewards.reward_feet_height_body,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=B2_FEET),
            "command_name": "base_velocity",
            "target_height": -0.4,
            "tanh_mult": 2.0,
        },
    )
    reward_feet_air_time = RewTerm(
        func=rewards.reward_feet_air_time,
        weight=0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=B2_FEET),
            "command_name": "base_velocity",
            "threshold": 0.12,
        },
    )
    reward_feet_contact_count = RewTerm(
        func=rewards.reward_feet_contact_count,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=B2_FEET),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )
    reward_feet_slide = RewTerm(
        func=rewards.reward_feet_slide,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=B2_FEET),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=B2_FEET),
        },
    )
    reward_feet_contact_without_cmd = RewTerm(
        func=rewards.reward_feet_contact_without_cmd,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=B2_FEET),
            "command_name": "base_velocity",
        },
    )
    reward_upward = RewTerm(
        func=rewards.reward_upward,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_upright_alive = RewTerm(
        func=rewards.reward_upright_alive,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_roll": 0.75,
            "max_pitch": 0.75,
            "min_height": 0.15,
        },
    )
    reward_fall_penalty = RewTerm(
        func=rewards.reward_fall_penalty,
        weight=-12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_roll": 1.0,
            "max_pitch": 1.0,
            "min_height": 0.0,
        },
    )
    reward_progress_to_goal = RewTerm(
        func=rewards.reward_progress_to_goal,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
            "clip": 0.10,
        },
    )
    reward_progress_from_start = RewTerm(
        func=rewards.reward_progress_from_start,
        weight=0.0,
        params={
            "parkour_name": "base_parkour",
            "clip": 0.10,
        },
    )
    reward_goal_reached = RewTerm(
        func=rewards.reward_goal_reached,
        weight=0.0,
        params={
            "parkour_name": "base_parkour",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.reward_tracking_goal_vel.func = rewards.reward_tracking_goal_vel_positive
        self.reward_tracking_goal_vel.weight = 0.0
        self.reward_tracking_yaw.func = rewards.reward_yaw_when_moving
        self.reward_tracking_yaw.weight = 0.0
        self.reward_collision.weight = -1.0
        self.reward_hip_pos.weight = 0.0
        self.reward_dof_error.weight = -1.0
        self.reward_action_rate.weight = -0.01
        self.reward_orientation.weight = 0.0
        self.reward_lin_vel_z.weight = -2.0
        self.reward_feet_stumble.weight = -1.0
        self.reward_torques.weight = -1.0e-5
        self.reward_dof_acc.weight = -1.0e-7
        self.reward_delta_torques.weight = -1.0e-7


@configclass
class B2EventCfg(EventCfg):
    def __post_init__(self):
        self.randomize_rigid_body_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=B2_BASE_BODY)
        self.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=B2_BASE_BODY)
        self.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names=B2_BASE_BODY)


@configclass
class B2ActionsCfg(ActionsCfg):
    def __post_init__(self):
        self.joint_pos.joint_names = B2_LEG_JOINTS
        self.joint_pos.preserve_order = True
        self.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.joint_pos.clip = {".*": (-100.0, 100.0)}


@configclass
class B2ProprioLidarTerminationsV2Cfg(TerminationsCfg):
    total_terminates = None
    time_out_or_goal = DoneTerm(
        func=terminations.parkour_time_out_or_goal,
        time_out=True,
    )
    fall = DoneTerm(
        func=terminations.parkour_fall_or_bad_orientation,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_roll": 1.5,
            "max_pitch": 1.5,
            "min_height": -0.25,
        },
    )

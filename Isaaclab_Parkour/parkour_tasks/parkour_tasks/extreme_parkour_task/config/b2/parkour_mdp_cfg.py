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
    reward_progress_to_goal = RewTerm(
        func=rewards.reward_progress_to_goal,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
            "clip": 0.25,
        },
    )
    reward_progress_from_start = RewTerm(
        func=rewards.reward_progress_from_start,
        weight=1.0,
        params={
            "parkour_name": "base_parkour",
            "clip": 0.25,
        },
    )
    reward_goal_reached = RewTerm(
        func=rewards.reward_goal_reached,
        weight=5.0,
        params={
            "parkour_name": "base_parkour",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.reward_tracking_goal_vel.func = rewards.reward_tracking_goal_vel_positive
        self.reward_tracking_goal_vel.weight = 3.0
        self.reward_tracking_yaw.func = rewards.reward_yaw_when_moving
        self.reward_tracking_yaw.weight = 0.25
        self.reward_collision.weight = -1.0
        self.reward_hip_pos.weight = -0.1
        self.reward_dof_error.weight = -0.01
        self.reward_action_rate.weight = -0.02
        self.reward_orientation.weight = -0.5
        self.reward_lin_vel_z.weight = -0.5
        self.reward_feet_stumble.weight = -0.5


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
        self.joint_pos.clip = {
            ".*_hip_joint": (-1.2, 1.2),
            ".*_thigh_joint": (-2.7, 3.5),
            ".*_calf_joint": (-2.7, -0.5),
        }


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

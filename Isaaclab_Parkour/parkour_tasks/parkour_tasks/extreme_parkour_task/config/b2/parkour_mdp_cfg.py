from isaaclab.managers import SceneEntityCfg
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
class B2TeacherRewardsCfg(TeacherRewardsCfg):
    def __post_init__(self):
        self.reward_collision.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[B2_BASE_BODY, ".*_calf", ".*_thigh"]
        )
        self.reward_feet_edge.params["asset_cfg"] = SceneEntityCfg(name="robot", body_names=B2_FEET)
        self.reward_feet_edge.params["base_body_name"] = B2_BASE_BODY


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

from isaaclab.utils import configclass

from .rsl_proprio_lidar_ppo_cfg import UnitreeB2ProprioLidarParkourPPORunnerCfg


@configclass
class UnitreeB2ProprioLidarParkourV2PPORunnerCfg(UnitreeB2ProprioLidarParkourPPORunnerCfg):
    max_iterations = 20000
    experiment_name = "unitree_b2_parkour_proprio_lidar_v2"

    def __post_init__(self):
        self.policy.init_noise_std = 1.0
        self.algorithm.learning_rate = 1.0e-3

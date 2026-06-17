"""B2 teacher-only extreme parkour locomotion tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="ATEC-Extreme-Parkour-Teacher-B2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeB2TeacherParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeB2ParkourTeacherPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Extreme-Parkour-Teacher-B2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeB2TeacherParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeB2ParkourTeacherPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Extreme-Parkour-Teacher-B2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeB2TeacherParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeB2ParkourTeacherPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Extreme-Parkour-ProprioLidar-B2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeB2ProprioLidarParkourEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_proprio_lidar_ppo_cfg:UnitreeB2ProprioLidarParkourPPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-Extreme-Parkour-ProprioLidar-B2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeB2ProprioLidarParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_proprio_lidar_ppo_cfg:UnitreeB2ProprioLidarParkourPPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-Extreme-Parkour-ProprioLidar-B2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeB2ProprioLidarParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_proprio_lidar_ppo_cfg:UnitreeB2ProprioLidarParkourPPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-Extreme-Parkour-ProprioLidar-B2-v2",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeB2ProprioLidarParkourEnvCfg_V2",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_proprio_lidar_ppo_v2_cfg:UnitreeB2ProprioLidarParkourV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-Extreme-Parkour-ProprioLidar-B2-Play-v2",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeB2ProprioLidarParkourEnvCfg_V2_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_proprio_lidar_ppo_v2_cfg:UnitreeB2ProprioLidarParkourV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-Extreme-Parkour-ProprioLidar-B2-Eval-v2",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeB2ProprioLidarParkourEnvCfg_V2_EVAL",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_proprio_lidar_ppo_v2_cfg:UnitreeB2ProprioLidarParkourV2PPORunnerCfg"
        ),
    },
)

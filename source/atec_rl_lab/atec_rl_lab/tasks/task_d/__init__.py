import gymnasium as gym

from .terrain import TASK_D_TERRAIN_CFG
from .env_cfg import TaskDEnvCfg, TaskDEnvB2Cfg, TaskDB2PiperLidarTeacherEnvCfg
from . import agents

# Ini juga biji
gym.register(
      id="ATEC-TaskD-B2Piper-LidarTeacher",
      entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
      disable_env_checker=True,
      kwargs={
          "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDB2PiperLidarTeacherEnvCfg",
          "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskDB2PiperLidarPPORunnerCfg",
      },
  )

gym.register(
    id = "ATEC-TaskD-G1",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvG1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskD-Tron1Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvTron1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskD-B2Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvB2Cfg",
    },
)

gym.register(
    id = "ATEC-TaskD-B2wPiper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvB2WCfg"
    },
)

__all__ = ['TaskDEnvCfg', 'TaskDEnvB2Cfg', 'TaskDB2PiperLidarTeacherEnvCfg']

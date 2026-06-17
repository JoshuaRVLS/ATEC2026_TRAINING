from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_teacher_ppo_cfg import (
    UnitreeGo2ParkourTeacherPPORunnerCfg,
)


@configclass
class UnitreeB2ParkourTeacherPPORunnerCfg(UnitreeGo2ParkourTeacherPPORunnerCfg):
    experiment_name = "unitree_b2_parkour_teacher"

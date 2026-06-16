# Created by skywoodsz on 4/4/26.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_x_greater_than(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_threshold: float = 2.0,
) -> torch.Tensor:
    """Terminate when robot root x in the local environment frame is greater than threshold."""
    robot = env.scene[asset_cfg.name]
    robot_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return robot_x > float(x_threshold)


from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.kit.app

from isaaclab.managers.command_manager import CommandTerm, CommandManager
from .parkour_manager_term_cfg import ParkourTermCfg

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

"""
Parkour Manager is dealing with goal heading position
It is similar to a CommandMangner which is a handling Position Command
"""

def sanitize_env_ids(env_ids: Sequence[int] | slice | None, num_envs: int, device: str | torch.device) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        valid_env_ids = env_ids.to(device=device, dtype=torch.long).flatten()
    else:
        valid_env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long).flatten()
    if valid_env_ids.numel() == 0:
        return valid_env_ids
    return valid_env_ids.clamp_(0, num_envs - 1)


class ParkourTerm(CommandTerm):
    def __init__(self, cfg: ParkourTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env) 

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        env_ids = sanitize_env_ids(env_ids, self.num_envs, self.device)
        if hasattr(self, "_sanitize_indices"):
            self._sanitize_indices()

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value.float()).item()
            if metric_value.ndim > 0 and metric_value.shape[0] == self.num_envs and env_ids.numel() > 0:
                metric_env_ids = env_ids.to(device=metric_value.device)
                metric_value[metric_env_ids] = 0.0

        self._resample(env_ids)
        if hasattr(self, "_sanitize_indices"):
            self._sanitize_indices()

        return extras
    
    def _resample(self, env_ids: Sequence[int]):
        if len(env_ids) != 0:
            self._resample_command(env_ids)

    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call to compute.
        """
        self._update_command()
        self._update_metrics()

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command generator has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code
    
    def __call__(self):
        pass 


    
class ParkourManager(CommandManager):
    _env: ParkourManagerBasedRLEnv
    def __init__(self, cfg: object, env: ParkourManagerBasedRLEnv):        
        super().__init__(cfg, env) 

    def __call__(self):
        for term in self._terms.values():
            term()

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, ParkourTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ParkourTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ParkourTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ParkourType.")
            # add class to dict
            self._terms[term_name] = term

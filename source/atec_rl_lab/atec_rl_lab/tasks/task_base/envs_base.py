# Created by skywoodsz on 2026/03/03.

import torch
from isaaclab.envs import ManagerBasedRLEnv

class BaseRLEnv(ManagerBasedRLEnv):
    def step(self, action: torch.Tensor):
        elapsed_time = self.episode_length_buf.to(torch.float32) * float(self.step_dt)

        obs, reward, terminated, truncated, info = super().step(action)

        early_done = (terminated | truncated) & (self.episode_length_buf <= 1)
        if torch.any(early_done) and not getattr(self, "_printed_early_done_debug", False):
            self._printed_early_done_debug = True
            env_id = int(torch.nonzero(early_done, as_tuple=False)[0, 0].item())
            print("[DEBUG early_done] env_id:", env_id)
            print("[DEBUG early_done] terminated:", bool(terminated[env_id].item()))
            print("[DEBUG early_done] truncated:", bool(truncated[env_id].item()))
            print("[DEBUG early_done] episode_length_buf:", int(self.episode_length_buf[env_id].item()))
            print("[DEBUG early_done] info keys:", sorted(info.keys()))
            log_info = info.get("log", {})
            print("[DEBUG early_done] log keys:", sorted(log_info.keys()) if isinstance(log_info, dict) else type(log_info))

        info["Elapsed_Time"] = elapsed_time
        info["Step_dt"] = float(self.step_dt)
        info["Episode_Length_s"] = float(self.max_episode_length_s)

        return obs, reward, terminated, truncated, info

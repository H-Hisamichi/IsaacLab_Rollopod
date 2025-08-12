from __future__ import annotations

import torch
from .manager_base import ManagerTermBase

class TraveledDistanceRecorder(ManagerTermBase):
    def __init__(self, config, env):
        super().__init__(config, env)
        self._env = env
        self._num_envs = env.num_envs
        self.device = env.device
        
        # Buffers to store positions and distances
        self._position_cache = [[] for _ in range(self._num_envs)]
        self._episode_travel_dist = torch.zeros(self._num_envs, device=self.device)

    def reset(self, env_ids):
        # Reset cache and distance for specified environments
        for idx in env_ids:
            self._position_cache[idx] = []
            self._episode_travel_dist[idx] = 0.0

    def pre_physics_step(self):
        # Record positions at the start of each physics step
        current_pos, _ = self._env.actors.get_world_poses()
        for i in range(self._num_envs):
            self._position_cache[i].append(current_pos[i].clone())

    def post_episode(self):
        # Calculate total distance at episode end
        for env_idx in range(self._num_envs):
            if len(self._position_cache[env_idx]) < 2:
                continue
                
            positions = torch.stack(self._position_cache[env_idx])
            deltas = positions[1:] - positions[:-1]
            distances = torch.norm(deltas, dim=1)
            self._episode_travel_dist[env_idx] = torch.sum(distances)

    def get_episode_travel_dist(self, env_ids=None):
        # Retrieve distances for curriculum manager
        if env_ids is None:
            return self._episode_travel_dist.clone()
        return self._episode_travel_dist[env_ids].clone()
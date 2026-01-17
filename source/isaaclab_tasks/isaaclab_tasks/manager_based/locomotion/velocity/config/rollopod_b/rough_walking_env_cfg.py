# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab.managers import SceneEntityCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.rollopod import ROLLOPOD_B_WALKING_CFG  # isort: skip


@configclass
class RollopodBRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to rollopod-b
        self.scene.robot = ROLLOPOD_B_WALKING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/MainBody"
        self.commands.base_velocity.debug_vis = False

        # events
        self.events.physics_material.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 0.9),#(0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.9),#(0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True
        }
        self.events.add_base_mass.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="MainBody"),
            "mass_distribution_params": (-3.0, 5.0),
            "operation": "add",
        }
        self.events.base_com.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="MainBody"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        }
        self.events.base_external_force_torque.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="MainBody"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        }

        # Rewards
        self.rewards.feet_air_time.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Toes"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        }
        self.rewards.undesired_contacts = None

        self.terminations.base_contact.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names="MainBody"), "threshold": 1.0}


@configclass
class RollopodBRoughEnvCfg_PLAY(RollopodBRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TraveledDistanceRecorder
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from isaaclab_assets.robots.rollopod import ROLLOPOD_B_ROLLING_CFG  # isort: skip

@configclass
class RollopodRewards(RewardsCfg):
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    ang_vel_xy_l2 = None
    feet_air_time = None
    undesired_contacts = None
    lin_vel_w_z_l2 = RewTerm(func=mdp.lin_vel_w_z_l2, weight=-4.0)
    steer_ang_vel_exp = RewTerm(
        func=mdp.steer_ang_vel_exp_2d, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(2.0)}
    )
    steer_ang_vel_exp_fine_grained = RewTerm(
        func=mdp.steer_ang_vel_exp_2d, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    track_com_ang_vel_z_exp = RewTerm(
        func=mdp.track_com_ang_vel_z_exp, weight=2.5, params={"command_name": "base_velocity", "std": math.sqrt(2.0)}
    )
    track_com_ang_vel_z_exp_fine_grained = RewTerm(
        func=mdp.track_com_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    # -- optional penalties
    flat_orientation_l2 = None
    flat_z_orientation_l2 = RewTerm(func=mdp.flat_z_orientation_l2, weight=-4.0)
    shake_rolling_penalty = RewTerm(
        func=mdp.shake_rolling_penalty, weight=-0.5, params={"command_name": "base_velocity", "scale": 0.5}
    )
    rolling_slip_penalty = RewTerm(
        func=mdp.rolling_slip_penalty_v2, weight=-0.2, params={ "scale": 0.6, "rolling_radius": 0.43}
    )


@configclass
class RollopodBRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: RollopodRewards = RollopodRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to rollopod-b
        self.scene.robot = ROLLOPOD_B_ROLLING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None

        self.commands.base_velocity = mdp.CamberAngleANDRollingVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            debug_vis=False,
            ranges=mdp.CamberAngleANDRollingVelocityCommandCfg.Ranges(
                angle_velocity=(-6.54, 6.54), camber_angle=(-1.0, 1.0), #rolling_radius=(0.35)
            ),
        )

        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale={
                ".*RevoluteJoint1": 0.25,
                ".*RevoluteJoint2": 0.2,
                ".*RevoluteJoint3": 0.2,
            },
            use_default_offset=True
        )

        self.observations.policy.base_lin_vel = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        self.observations.policy.base_ang_vel = ObsTerm(
            func=mdp.base_com_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        self.observations.policy.height_scan = None

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
        self.events.reset_base.params = {
            "pose_range": {"yaw": (-3.14, 3.14)}, # def: (-3.14, 3.14)
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints = None
        self.events.push_robot.params = {"velocity_range": {"yaw": (-0.5, 0.5)}}

        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.dof_acc_l2.weight = -4.0e-7
        self.rewards.action_rate_l2.weight = -0.005

        self.terminations.base_contact.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names="MainBody"), "threshold": 1.0}

        self.curriculum.terrain_levels.func = TraveledDistanceRecorder

        



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

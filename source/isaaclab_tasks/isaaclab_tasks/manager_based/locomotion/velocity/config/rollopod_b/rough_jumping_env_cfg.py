# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, CurriculumCfg, TerminationsCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
#from isaaclab.managers import TraveledDistanceRecorder
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

##
# Pre-defined configs
##
from isaaclab_assets.robots.rollopod import ROLLOPOD_B_JUMPING_CFG  # isort: skip

@configclass
class JumpingRewards(RewardsCfg):
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    feet_air_time = None
    undesired_contacts = None
    lin_vel_z_l2 = None
    #jump_success_terminal_latched = RewTerm(
    #    func=mdp.jump_success_terminal_latched, weight=1.0, params={"command_name": "base_velocity"}
    #)

    #track_pos_exp = RewTerm(
    #    func=mdp.track_pos_exp, weight=0.7, params={"command_name": "base_velocity", "std": math.sqrt(3.0)}
    #)
    #track_pos_exp_fine_grained = RewTerm(
    #    func=mdp.track_pos_exp, weight=0.3, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    #)

    track_lin_vel_z_exp = RewTerm(
        func=mdp.track_lin_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(4.0)}
    )
    track_xy_pos_exp = RewTerm(
        func=mdp.track_xy_pos_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(2.0)}
    )
    #track_pos_w_exp_fine_grained = RewTerm(
    #    func=mdp.track_pos_exp_latched_v4, weight=0.6, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    #)
    #track_pos_exp_v2 = RewTerm(
    #    func=mdp.track_pos_exp_v2, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25), "std_high": math.sqrt(0.1)}
    #)
    #track_pos_binary = RewTerm(
    #    func=mdp.track_pos_binary, weight=1.0, params={"command_name": "base_velocity", "threshold": 0.05}
    #)
    #jump_vel_w_z_l2 = RewTerm(func=mdp.jump_vel_w_z_l2, weight=0.5)
    # -- optional penalties
    #penalty_overshoot_height = RewTerm(
    #    func=mdp.penalty_overshoot_height, weight=-0.0, params={"command_name": "base_velocity", "threshold": 0.1}
    #)
    ang_vel_xyz_l2 = RewTerm(func=mdp.ang_vel_xyz_l2, weight=-0.08)
    #penalty_overshoot_height_fine_grained = RewTerm(
    #    func=mdp.penalty_overshoot_height, weight=-0.6, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    #)

@configclass
class JumpingCurriculums(CurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

@configclass
class JumpingTerminationsCfg(TerminationsCfg):
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="MainBody"), "threshold": 1.0},
    )
    #landing_after_jump = DoneTerm(
    #    func=mdp.landing_after_jump_v2,
    #    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 1.0},
    #)

@configclass
class RollopodBRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: JumpingRewards = JumpingRewards()
    curriculum: JumpingCurriculums = JumpingCurriculums()
    terminations: JumpingTerminationsCfg = JumpingTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        #self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.gravity = (0.0, 0.0, -1.62)
        # switch robot to rollopod-b
        self.scene.robot = ROLLOPOD_B_JUMPING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/MainBody"
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0))
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.2, size=[1.0, 1.0])

        self.commands.base_velocity = mdp.JumpingCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            #rel_standing_envs=0.02,
            debug_vis=False,
            ranges=mdp.JumpingCommandCfg.Ranges(
                pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), lin_vel_z=(3.0, 5.7)
            ),
        )

        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale={
                ".*RevoluteJoint1": 0.3,
                ".*RevoluteJoint2": 0.3,
                ".*RevoluteJoint3": 0.3,
            },
            use_default_offset=True
        )

        # observations
        #self.observations.policy.base_lin_vel = ObsTerm(
        #    func=mdp.base_com_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        #)
        #self.observations.policy.base_ang_vel = ObsTerm(
        #    func=mdp.base_com_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        #)
        #self.observations.policy.height_scan = None
        #self.observations.policy.base_lin_vel.func = mdp.root_lin_vel_w

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
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        }
        self.events.base_com.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="MainBody"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        }
        self.events.base_external_force_torque.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="MainBody"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        }
        self.events.reset_base.params = {
            "pose_range": {"yaw": (0.0, 0.0)}, # def: (-3.14, 3.14)
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
        self.events.push_robot = None

        # Rewards
        self.rewards.dof_torques_l2.weight = -0.1e-5
        self.rewards.ang_vel_xy_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -0.1

        



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
        self.commands.base_velocity.debug_vis = True
        self.scene.height_scanner.debug_vis = True

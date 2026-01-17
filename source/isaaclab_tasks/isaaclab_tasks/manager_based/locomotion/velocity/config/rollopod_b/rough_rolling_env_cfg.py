# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    CurriculumCfg,
    ObservationsCfg
)
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
#from isaaclab.managers import TraveledDistanceRecorder
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

##
# Pre-defined configs
##
from isaaclab_assets.robots.rollopod import ROLLOPOD_B_ROLLING_CFG  # isort: skip
from isaaclab.terrains.config.rollopod_rough import ROUGH_TERRAINS_CFG_TEST  # isort: skip

@configclass
class RollopodRewards(RewardsCfg):
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    ang_vel_xy_l2 = None
    feet_air_time = None
    undesired_contacts = None
    lin_vel_z_l2 = None
    lin_vel_w_z_l2 = RewTerm(func=mdp.lin_vel_w_z_l2, weight=-0.6)
    #rolling_ang_vel = RewTerm(func=mdp.rolling_ang_vel, weight=0.1, params={"command_name": "base_velocity"})
    track_lin_vel_xy_w_exp = RewTerm(
        func=mdp.track_lin_vel_dir_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(1.0)}
    )
    #track_lin_vel_xy_w_exp_fine_grained = RewTerm(
    #    func=mdp.track_lin_vel_dir_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(1.0)}
    #)
    track_com_ang_vel_z_exp = RewTerm(
        func=mdp.track_com_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(1.0)}
    )
    #track_com_ang_vel_z_exp_fine_grained = RewTerm(
    #    func=mdp.track_com_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(1.0)}
    #)
    # -- optional penalties
    flat_orientation_l2 = None
    #flat_z_orientation_l2 = RewTerm(func=mdp.flat_z_orientation_l2, weight=-0.25)
    shake_rolling_penalty = RewTerm(
        func=mdp.shake_rolling_penalty, weight=-0.5, params={"command_name": "base_velocity", "scale": 0.4}
    ) # RL 2 phase weight = -0.05
    rolling_slip_penalty = RewTerm(
        func=mdp.rolling_slip_penalty, weight=-0.1, params={"command_name": "base_velocity", "scale": 0.5, "rolling_radius": 0.33}
    ) # RL 2 phase weight = -0.01
    #shake_rolling_penalty = RewTerm(func=mdp.ang_acc_w_z_l2, weight=-0.0001, params={"target_body": "MainBody"})
    lin_vel_z_penalty = RewTerm(func=mdp.lin_vel_z_penalty, weight=-0.5) # RL 2 phase weight = -0.05

#@configclass
#class RollopodObservations(ObservationsCfg):
#    """Observation specifications for the MDP."""
#    @configclass
#    class RollopodPolicyCfg(ObservationsCfg.PolicyCfg):
#        """Observations for policy group."""
#        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.1, n_max=0.1))
    
#    policy: RollopodPolicyCfg = RollopodPolicyCfg()

@configclass
class RollopodCurriculums(CurriculumCfg):
    terrain_levels = None
    # transition from lenient to strict reward evaluation
    #track_lin_vel_xy_w_exp_weight = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "track_lin_vel_xy_w_exp", "weight": 0.0, "num_steps": 2000}
    #)
    #track_lin_vel_xy_w_exp_fine_grained_weight = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "track_lin_vel_xy_w_exp_fine_grained", "weight": 1.0, "num_steps": 2000}
    #)
    #track_com_ang_vel_z_exp_weight = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "track_com_ang_vel_z_exp", "weight": 0.0, "num_steps": 2000}
    #)
    #track_com_ang_vel_z_exp_fine_grained_weight = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "track_com_ang_vel_z_exp_fine_grained", "weight": 1.0, "num_steps": 2000}
    #)
    # gradual relaxation of penalty weights
    #shake_rolling_penalty_weight_f1 = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "shake_rolling_penalty", "weight": -0.35, "num_steps": 2000}
    #)
    #rolling_slip_penalty_weight_f1 = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "rolling_slip_penalty", "weight": -0.085, "num_steps": 3000}
    #)
    #lin_vel_z_penalty_weight_f1 = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "lin_vel_z_penalty", "weight": -0.35, "num_steps": 2000}
    #)
    #shake_rolling_penalty_weight_f2 = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "shake_rolling_penalty", "weight": -0.25, "num_steps": 3000}
    #)
    #rolling_slip_penalty_weight_f2 = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "rolling_slip_penalty", "weight": -0.05, "num_steps": 3500}
    #)
    #lin_vel_z_penalty_weight_f2 = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "lin_vel_z_penalty", "weight": -0.25, "num_steps": 3000}
    #)
    


@configclass
class RollopodBRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    #observations: RollopodObservations = RollopodObservations()
    rewards: RollopodRewards = RollopodRewards()
    curriculum: RollopodCurriculums = RollopodCurriculums()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # simulation settings
        self.sim.gravity = (0.0, 0.0, -1.62)
        # switch robot to rollopod-b
        # scene
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_TEST
        self.scene.terrain.max_init_terrain_level = None
        self.scene.robot = ROLLOPOD_B_ROLLING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/MainBody"
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.165))
        self.scene.height_scanner.ray_alignment = "world"
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.5, size=[3.5, 3.5])
        #self.scene.height_scanner.debug_vis = True
        #self.scene.height_scanner = None

        self.commands.base_velocity = mdp.UniformWorldVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            debug_vis=False,
            ranges=mdp.UniformWorldVelocityCommandCfg.Ranges(
                rolling_speed=(-8.46, 8.46), heading=(-math.pi, math.pi)
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
        self.observations.policy.height_scan.clip = (0.0, 4.0)
        #self.observations.policy.height_scan = None
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

        # Rewards
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01

        self.terminations.base_contact.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names="MainBody"), "threshold": 1.0}

        #self.curriculum.terrain_levels.func = mdp.terrain_levels_vel_rollopod

        



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

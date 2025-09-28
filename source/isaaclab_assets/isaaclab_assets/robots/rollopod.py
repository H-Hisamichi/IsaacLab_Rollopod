# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mujoco Ant robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, ActuatorNetLSTMCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration - Actuators.
##

ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
)
"""Configuration for ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model."""

##
# Configuration
##

ROLLOPOD_B_ROLLING_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/robot/hisamichi/IsaacSim/rollopod_b_simplification_v29.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            #sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        #copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        rot=(0.7071, -0.7071, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    },
)

ROLLOPOD_B_WALKING_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        #usd_path="/home/robot/hisamichi/IsaacSim/rollopod_b_simplification_v26_quadruped.usd",
        usd_path="/home/robot/hisamichi/IsaacSim/rollopod_b_simplification_v29.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            #sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        #rot=(0.7071, -0.7071, 0.0, 0.0),
        joint_pos={
            "Leg1_RevoluteJoint1":0.0, "Leg2_RevoluteJoint1":0.0, "Leg3_RevoluteJoint1":0.0, "Leg4_RevoluteJoint1":0.0, "Leg5_RevoluteJoint1":0.0, "Leg6_RevoluteJoint1":0.0,
            "Leg1_RevoluteJoint2":2.01, "Leg2_RevoluteJoint2":2.01, "Leg3_RevoluteJoint2":2.01, "Leg4_RevoluteJoint2":2.01, "Leg5_RevoluteJoint2":2.01, "Leg6_RevoluteJoint2":2.01,
            "Leg1_RevoluteJoint3":-2.495, "Leg2_RevoluteJoint3":-2.495, "Leg3_RevoluteJoint3":-2.495, "Leg4_RevoluteJoint3":-2.495, "Leg5_RevoluteJoint3":-2.495, "Leg6_RevoluteJoint3":-2.495
        }, # def: 0.0, 1.71, -2.195
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    },
)
"""Configuration for the Mujoco Ant robot."""

ROLLOPOD_B_ROLLING_CFG_V2 = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/robot/hisamichi/IsaacSim/rollopod_b_simplification_v26.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        rot=(0.7071, -0.7071, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    #soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=40, # 23.5
            saturation_effort=40, # 23.5
            velocity_limit=30.0,
            stiffness=None,
            damping=None,
            friction=0.0,
        ),
    },
)

ROLLOPOD_B_ROLLING_CFG_V3 = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/robot/hisamichi/IsaacSim/rollopod_b_simplification_v26.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        rot=(0.7071, -0.7071, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    #soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-C robot using actuator-net."""

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

CARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Robots/Carter/carter_v1.usd
        usd_path=f"omniverse://localhost/Users/carter_v1.usd",
        # usd_path=f"omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Robots/Carter/carter_v1.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), joint_pos={"left_wheel": 0.0, "right_wheel": 0.0}
    ),
    actuators={
        # "left_actuator": ImplicitActuatorCfg(
        #     joint_names_expr=["left_wheel"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
        # "right_actuator": ImplicitActuatorCfg(
        #     joint_names_expr=["right_wheel"], 
        #     effort_limit=400.0, 
        #     velocity_limit=100.0, 
        #     stiffness=0.0, 
        #     damping=10.0
        # ),
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=100.0, 
            velocity_limit=10.0, 
            stiffness=0.0,
            damping=10.0,
        ),
    },
)
"""Configuration for a simple Carter robot."""

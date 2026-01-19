# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kuka-lbr-iiwa arm robots and Allegro Hand.

The following configurations are available:

* :obj:`KUKA_ALLEGRO_CFG`: Kuka Allegro with implicit actuator model.

Reference:

* https://www.kuka.com/en-us/products/robotics-systems/industrial-robots/lbr-iiwa
* https://www.wonikrobotics.com/robot-hand

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

KUKA_7DOF_THREE_FINGER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/usd/assemble_right_hand_robot/assemble_right_hand_robot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "iiwa7_joint_1": 0.0,
            "iiwa7_joint_2": 0.0,
            "iiwa7_joint_3": 0.7854,
            "iiwa7_joint_4": 1.5708,
            "iiwa7_joint_5": -1.5708,
            "iiwa7_joint_6": -1.5708,
            "iiwa7_joint_7": 0.0,
            "index_joint1": 0.0,
            "middle_joint1": 0.0,
            "thumb_joint1": 1.7,
            "index_joint2": -0.0,
            "middle_joint2": -0.0,
            "thumb_joint2": 0.0,
            "thumb_joint3": -0.0,
        },
    ),
    actuators={
        "kuka_allegro_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "iiwa7_joint_1",
                "iiwa7_joint_2",
                "iiwa7_joint_3",
                "iiwa7_joint_4",
                "iiwa7_joint_5",
                "iiwa7_joint_6",
                "iiwa7_joint_7",
                "index_joint1",
                "middle_joint1",
                "thumb_joint1",
                "index_joint2",
                "middle_joint2",
                "thumb_joint2",
                "thumb_joint3",
            ],
            effort_limit_sim={
                "iiwa7_joint_1": 300.0,
                "iiwa7_joint_2": 300.0,
                "iiwa7_joint_3": 300.0,
                "iiwa7_joint_4": 300.0,
                "iiwa7_joint_5": 300.0,
                "iiwa7_joint_6": 300.0,
                "iiwa7_joint_7": 300.0,
                "index_joint1": 0.8,
                "middle_joint1": 0.8,
                "thumb_joint1": 0.8,
                "index_joint2": 0.8,
                "middle_joint2": 0.8,
                "thumb_joint2": 0.8,
                "thumb_joint3": 0.8,
            },
            stiffness={
                "iiwa7_joint_1": 300.0,
                "iiwa7_joint_2": 300.0,
                "iiwa7_joint_3": 300.0,
                "iiwa7_joint_4": 300.0,
                "iiwa7_joint_5": 100.0,
                "iiwa7_joint_6": 50.0,
                "iiwa7_joint_7": 25.0,
                "index_joint1": 3.0,
                "middle_joint1": 3.0,
                "thumb_joint1": 3.0,
                "index_joint2": 3.0,
                "middle_joint2": 3.0,
                "thumb_joint2": 3.0,
                "thumb_joint3": 3.0,
            },
            damping={
                "iiwa7_joint_1": 45.0,
                "iiwa7_joint_2": 45.0,
                "iiwa7_joint_3": 45.0,
                "iiwa7_joint_4": 45.0,
                "iiwa7_joint_5": 20.0,
                "iiwa7_joint_6": 15.0,
                "iiwa7_joint_7": 15.0,
                "index_joint1": 0.1,
                "middle_joint1": 0.1,
                "thumb_joint1": 0.1,
                "index_joint2": 0.1,
                "middle_joint2": 0.1,
                "thumb_joint2": 0.1,
                "thumb_joint3": 0.1,
            },
            friction={
                "iiwa7_joint_1": 1.0,
                "iiwa7_joint_2": 1.0,
                "iiwa7_joint_3": 1.0,
                "iiwa7_joint_4": 1.0,
                "iiwa7_joint_5": 1.0,
                "iiwa7_joint_6": 1.0,
                "iiwa7_joint_7": 1.0,
                "index_joint1": 0.01,
                "middle_joint1": 0.01,
                "thumb_joint1": 0.01,
                "index_joint2": 0.01,
                "middle_joint2": 0.01,
                "thumb_joint2": 0.01,
                "thumb_joint3": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the Kuka Self Three Finger robot and drives its gripper joints.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/test_kuka_self.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates the Kuka Self Three Finger robot.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets import KUKA_7DOF_THREE_FINGER_CFG


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Origin 1 with Kuka Self Three Finger
    origins = define_origins(num_origins=1, spacing=2.0)
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    
    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    )
    cfg.func("/World/Origin1/Table", cfg, translation=(0.0, 0.0, 1.03))
    
    # -- Robot
    robot_cfg = KUKA_7DOF_THREE_FINGER_CFG.replace(prim_path="/World/Origin1/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 1.03)
    robot = Articulation(cfg=robot_cfg)

    # return the scene information
    scene_entities = {
        "kuka_self": robot,
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    kuka_robot = entities["kuka_self"]
    
    # Get indices for finger joints to drive them specifically
    # Based on the config, finger joints usually start with 'finger_joint'
    # We will find them dynamically
    joint_names = kuka_robot.data.joint_names
    # Update filter to include new joint names (index, middle, thumb) as well as legacy 'finger'
    finger_keywords = ["finger", "index", "middle", "thumb"]
    finger_joint_indices = [i for i, name in enumerate(joint_names) if any(sub in name for sub in finger_keywords)]
    print(f"[INFO]: Finger joint indices: {finger_joint_indices}")
    print(f"[INFO]: Finger joint names: {[joint_names[i] for i in finger_joint_indices]}")

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:  # Reset every 500 steps
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            for index, robot in enumerate(entities.values()):
                # root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # set joint positions
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robots state...")

        # Drive finger joints with a large sine wave movement
        # Frequency 0.5 Hz, Amplitude 1.0 rad
        sine_wave = 1.0 * np.sin(2 * np.pi * 0.5 * sim_time)
        
        # We want to move fingers from 0 to some positive/negative value
        # Let's target the default position + offset
        current_target = kuka_robot.data.default_joint_pos.clone()
        
        # Apply movement to finger joints only
        for joint_idx in finger_joint_indices:
             current_target[:, joint_idx] += sine_wave

        # Clamp to limits
        current_target = current_target.clamp_(
            kuka_robot.data.soft_joint_pos_limits[..., 0], kuka_robot.data.soft_joint_pos_limits[..., 1]
        )
        
        # apply action to the robot
        kuka_robot.set_joint_position_target(current_target)
        kuka_robot.write_data_to_sim()

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera to look close at the gripper
    sim.set_camera_view([1.5, 0.0, 1.5], [0.0, 0.0, 1.03])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

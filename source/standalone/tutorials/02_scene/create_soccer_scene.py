# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/03_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
##
# Pre-defined configs
##
from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip
from omni.isaac.lab_assets import CARTER_CFG
import omni.isaac.lab.utils.math as math_utils
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    # cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    carter: ArticulationCfg = CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    gate = AssetBaseCfg (
        prim_path="{ENV_REGEX_NS}/gate",
        spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Users/gate.usd",)
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["carter"]
    ball = scene["ball"]
    robot._left_dof_idx, _ = robot.find_joints("left_wheel")
    robot._right_dof_idx, _ = robot.find_joints("right_wheel")
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 1000 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_vel += torch.rand_like(joint_vel) * 1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
            # print(joint_vel)
        # Apply random action
        # -- generate random joint efforts
        # print("robot.data.joint_vel",robot.data.joint_vel)
        # robot_vel_left = robot.data.joint_vel[:, robot._left_dof_idx[0]]
        # print("robot_vel_left",robot_vel_left)
        # robot_vel_right = robot.data.joint_vel[:, robot._right_dof_idx[0]]
        # print("robot_vel_right",robot_vel_right)
        # robot_vel = torch.stack([robot_vel_left, robot_vel_right], dim=-1)
        # print("robot_vel",robot_vel)
        robot_state = robot.data.root_state_w
        ball_state = ball.data.root_state_w
        ball_pos = ball_state[:, :3]
        robot_eular = math_utils.euler_xyz_from_quat(robot_state[:, 3:7])
        # print("robot_eular",robot_eular)
        robot_eular = torch.stack(robot_eular, dim=-1)
        # print("robot_state",robot_state)
        # print("robot_eular",robot_eular)
        # print("ball_pos",ball_pos)
        # print("scene.env_origins",scene.env_origins)
        rela_ball_pos = ball_pos - scene.env_origins
        # print("rela_ball_pos",rela_ball_pos)
        robot_vel = robot.data.joint_vel[:, [robot._left_dof_idx[0], robot._right_dof_idx[0]]]
        efforts = torch.randn_like(robot_vel) * 50.0
        # print(efforts)
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts, joint_ids=[robot._left_dof_idx[0], robot._right_dof_idx[0]])
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=22.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

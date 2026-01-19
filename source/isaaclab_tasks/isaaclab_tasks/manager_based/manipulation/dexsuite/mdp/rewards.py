# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w

    # [DEBUG] Print link names and sample data
    # Printing only periodically to avoid console flooding
    # if env.common_step_counter % 50 == 0:
    #     names = [asset.body_names[i] for i in asset_cfg.body_ids]
    #     print(f"\n[DEBUG] object_ee_distance (Step {env.common_step_counter})")
    #     print(f"  > Asset Name: {asset_cfg.name}")
    #     print(f"  > Tracking Links: {names}")
    #     print(f"  > Values (Env 0):")
    #     print(f"    - Asset Pos: {asset_pos[0].cpu().tolist()}")
    #     print(f"    - Object Pos: {object_pos[0].cpu().tolist()}")

    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1 - torch.tanh(object_ee_distance / std)


def contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""

    # Determine which set of sensors is available in the environment
    if "Empty_Link1_3_object_s" in env.scene.sensors.keys():
        # Kuka 7DOF Three Finger names
        thumb_name = "Empty_Link1_3_object_s"
        index_name = "Empty_Link2_1_object_s"
        middle_name = "Empty_Link3_1_object_s"
    elif "Empty_Link1_4_object_s" in env.scene.sensors.keys():
        # Kuka Self Three Finger names
        thumb_name = "Empty_Link1_4_object_s"
        index_name = "Empty_Link2_3_object_s"
        middle_name = "Empty_Link3_3_object_s"
    else:
        # Fallback or other robot (e.g. Allegro original)
        # Note: Adjust these keys if using original Allegro
        thumb_name = "thumb_link_3_object_s"
        index_name = "index_link_3_object_s"
        middle_name = "middle_link_3_object_s"

    thumb_contact_sensor: ContactSensor = env.scene.sensors[thumb_name]
    index_contact_sensor: ContactSensor = env.scene.sensors[index_name]
    middle_contact_sensor: ContactSensor = env.scene.sensors[middle_name]

    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)

    good_contact_cond1 = (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold)
    )

    return good_contact_cond1

def contacts_7DOF(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""

    # thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_link_3_object_s"]
    # index_contact_sensor: ContactSensor = env.scene.sensors["index_link_3_object_s"]
    # middle_contact_sensor: ContactSensor = env.scene.sensors["middle_link_3_object_s"]
    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_tip_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_tip_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_tip_object_s"]
    thumb_link_contact_sensor: ContactSensor = env.scene.sensors["Empty_Link1_2_object_s"]
    index_link_contact_sensor: ContactSensor = env.scene.sensors["Empty_Link2_object_s"]
    middle_link_contact_sensor: ContactSensor = env.scene.sensors["Empty_Link3_object_s"]
    # ring_contact_sensor: ContactSensor = env.scene.sensors["ring_link_3_object_s"]
    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    thumb_link_contact = thumb_link_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_link_contact = index_link_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_link_contact = middle_link_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    # ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    thumb_link_contact_mag = torch.norm(thumb_link_contact, dim=-1)
    index_link_contact_mag = torch.norm(index_link_contact, dim=-1)
    middle_link_contact_mag = torch.norm(middle_link_contact, dim=-1)
    # ring_contact_mag = torch.norm(ring_contact, dim=-1)
    good_contact_cond1 = ((thumb_contact_mag > threshold)| (thumb_link_contact_mag > threshold)) & (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (middle_link_contact_mag > threshold) | (index_link_contact_mag > threshold)
    )
    # good_contact_cond1 = (thumb_contact_mag > threshold) | (middle_contact_mag > threshold) | (index_contact_mag > threshold)

    return good_contact_cond1

def reward_thumb_open(env, target: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["thumb_joint1"])) -> torch.Tensor:
    """Reward for keeping the thumb joint open (close to a target value).

    The reward is tracked using a tanh kernel on the distance from the target value.
    """
    # extract the used quantities (states)
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    # calculate the distance to the target value
    distance = torch.abs(joint_pos - target).sum(dim=-1)
    # apply tanh kernel
    return 1.0 - torch.tanh(distance / std)

def reward_grasp_status(
    env: ManagerBasedRLEnv, 
    std: float, 
    targets: dict[str, float], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for keeping the hand in a specific grasping posture (joint angles).

    The reward is tracked using a tanh kernel on the L2 distance from the target configuration.
    """
    # extract the used quantities (states)
    robot = env.scene[asset_cfg.name]
    
    # Resolve joint indices dynamically. 
    # We assume the robot joint names don't change at runtime.
    all_joint_names = robot.data.joint_names
    
    # Build lists for indices and values to ensure order matches
    joint_indices = []
    target_values = []
    
    for joint_name, target_val in targets.items():
        if joint_name in all_joint_names:
            joint_indices.append(all_joint_names.index(joint_name))
            target_values.append(target_val)
            
    # if not joint_indices:
    #     # [DEBUG]
    #     if env.common_step_counter % 50 == 0:
    #         print(f"\n[DEBUG] grasp_posture_reward (Step {env.common_step_counter}) - NO JOINTS FOUND")
    #         print(f"  > Targets keys: {list(targets.keys())}")
    #         print(f"  > Robot joint names sample: {all_joint_names[:5]}...")
    #     return torch.zeros(env.num_envs, device=env.device)
        
    # Convert targets to tensor
    target_tensor = torch.tensor(target_values, device=env.device)
    
    # Get current joint positions for the specific joints
    # shape: (num_envs, num_tracked_joints)
    current_pos = robot.data.joint_pos[:, joint_indices]
    
    # Calculate L2 distance per environment
    distance = torch.norm(current_pos - target_tensor, dim=-1)

    # [DEBUG]
    # if env.common_step_counter % 50 == 0:
    #     print(f"\n[DEBUG] grasp_posture_reward (Step {env.common_step_counter})")
    #     missed = [k for k in targets.keys() if k not in all_joint_names]
    #     if missed:
    #         print(f"  > WARN: Missed joints: {missed}")
    #     print(f"  > Matched {len(joint_indices)} joints.")
    #     print(f"  > Sample Env 0 Current Pos: {current_pos[0].detach().cpu().numpy()}")
    #     print(f"  > Target Values: {target_values}")
    #     print(f"  > Distance (Env 0): {distance[0].item()}")
    #     rew_val = 1.0 - torch.tanh(distance / std)
    #     print(f"  > Reward raw val (Env 0): {rew_val[0].item()}")
    
    # Apply tanh kernel
    return 1.0 - torch.tanh(distance / std)

def reward_palm_down(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize if the palm Y-axis is pointing upwards (Z > threshold).

    This assumes the palm's orientation is such that local Y-axis represents the 'up' vector of the palm surface.
    We want to discourage this vector from aligning with world +Z.
    """
    # Extract the robot/palm body state
    robot = env.scene[asset_cfg.name]
    
    # Get the quaternion of the palm/base_link relative to world
    # Assuming asset_cfg points to the specific body, or we use root if it's the robot root
    # If asset_cfg.body_names is set, resolve body ID dynamically
    if asset_cfg.body_names is not None:
        # resolve ids using find_bodies
        if isinstance(asset_cfg.body_names, str):
            body_names = [asset_cfg.body_names]
        else:
            body_names = asset_cfg.body_names
        
        ids, _ = robot.find_bodies(body_names)
        
        if len(ids) > 0:
             palm_quat = robot.data.body_quat_w[:, ids[0]]
        else:
             palm_quat = robot.data.root_quat_w
    elif asset_cfg.body_ids is not None:
        # Fallback if names not provided but IDs are
        if isinstance(asset_cfg.body_ids, (list, tuple)):
             if len(asset_cfg.body_ids) > 0:
                 idx = asset_cfg.body_ids[0]
                 palm_quat = robot.data.body_quat_w[:, idx]
             else:
                 palm_quat = robot.data.root_quat_w
        else:
             # integer case
             palm_quat = robot.data.body_quat_w[:, asset_cfg.body_ids]
    else:
        # Fallback to root if body not specified
        palm_quat = robot.data.root_quat_w

    # The local Y-axis vector is [0, 1, 0]
    # Rotate this vector by the palm's quaternion to get it in world frame
    # math_utils.quat_rotate(q, v) rotates vector v by quaternion q
    local_y = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    world_y = math_utils.quat_rotate(palm_quat, local_y)
    
    # Check the Z component of this world vector
    # If world_y.z > threshold (e.g. 0.0), it means it's pointing somewhat up
    # We want to penalize this.
    # Reward = 0.0 if z > threshold (bad), 1.0 if z <= threshold (good)
    # OR continuous penalty based on Z value
    
    # Let's map Z from [-1, 1] to a reward.
    # We want Z to be negative (pointing down).
    # Simple strategy: Reward = 1 if Z < -0.5, else decreasing.
    # Or simply return -Z (maximize negative Z -> minimize Z -> point down)
    
    # [Mod] Only penalize if pointing UP (Z > threshold). 
    # If Pointing DOWN (Z < threshold), reward is 0 (neutral).
    # Since we want a "penalty", we return a negative value when Z > threshold.
    
    # world_y[:, 2] is the Z component.
    # If Z > threshold (e.g. 0.0), it is pointing up. We want to return negative value.
    # We can return -Z directly if Z > 0, else 0.
    
    z_component = world_y[:, 2]
    # Use torch.clamp or where to filter
    # If z > threshold, penalty = -z (or similar scaling). If z <= threshold, penalty = 0.
    
    return torch.where(z_component > threshold, -z_component, torch.zeros_like(z_component))

def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    if not rot_std:
        # square is not necessary but this help to keep the final value between having rot_std or not roughly the same
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = torch.norm(rot_err, dim=1)
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, 1.0).float()

def position_command_error_tanh_7DOF(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts_7DOF(env, 1.0).float()

def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)

    return (1 - torch.tanh(quat_distance / std)) * contacts(env, 1.0).float()

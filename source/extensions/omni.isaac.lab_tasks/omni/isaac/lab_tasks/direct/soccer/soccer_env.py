# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, spawn_from_usd
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform

from omni.isaac.lab_assets.soccer import CARTER_CFG
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
@configclass
class SoccerEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_scale = 30.0  # [N]
    action_bias = 0.0
    num_actions = 2
    num_observations = 33
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    # robot_cfg: ArticulationCfg = Carter_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    carter: ArticulationCfg = CARTER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    left_dof_name = "left_wheel"
    right_dof_name = "right_wheel"

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=18.0, replicate_physics=True)

    # reset
    max_ball_pos = [7.0, 7.0]
    initial_carter_range = [3.0, 4.0]
    initial_ball_range = [3.0, 2.0]
    # reward scales
    rew_scale_ang_vel_xy = 0.0
    rew_scale_stop = -0.0
    rew_scale_out_of_bounds = 0.0
    rew_scale_action_rate = 0.0
    rew_scale_vel_dir = 1.0
    rew_scale_speed = 0.0
    # reward scales
    lin_vel_reward_scale = 30.0
    # yaw_rate_reward_scale = 0.5
    heading_reward_scale = 50.0
    distance_reward_scale = 100.0
    out_of_bounds_reward_scale = -10.0
    ball_dis_reward_scale = 1e4
    ball_heading_reward_scale = 100.0
    ball_in_gate_reward_scale = 1e6
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-6
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = -5.0
    # hyperparameters
    position_target_sigma_soft = 1.0

class SoccerEnv(DirectRLEnv):
    cfg: SoccerEnvCfg

    def __init__(self, cfg: SoccerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        self._left_dof_idx, _ = self.carter.find_joints(self.cfg.left_dof_name)
        self._right_dof_idx, _ = self.carter.find_joints(self.cfg.right_dof_name)
        self.left_idx = self._left_dof_idx[0]
        self.right_idx = self._right_dof_idx[0]
        self.action_scale = self.cfg.action_scale

        self._actions_joints = torch.zeros(self.num_envs, 2, device=self.device)

        # command
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                # "track_ang_vel_z_exp",
                "track_heading",
                "track_distance",
                "out_of_bounds",
                "ball_dis",
                "ball_heading",
                "ball_in_gate",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "flat_orientation_l2",
            ]
        }
        self.carter_state = self.carter.data.root_state_w
        self.ball_state = self.ball.data.root_state_w
        # carter
        self.base_lin_vel = self.carter.data.root_lin_vel_w
        self.base_ang_vel = self.carter.data.root_ang_vel_w
        self.base_quat = self.carter.data.root_quat_w
        self.base_pos = self.carter.data.root_pos_w
        self.root_lin_vel_b = self.carter.data.root_lin_vel_b
        self.root_ang_vel_b = self.carter.data.root_ang_vel_b
        self.joint_vel = self.carter.data.joint_vel
        # ball
        self.ball_pos = self.ball.data.root_pos_w
        self.ball_lin_vel = self.ball.data.root_lin_vel_w

        self.gate_x = torch.tensor([6.0, 6.8], device=self.device)
        self.gate_y = torch.tensor([-1.0, 1.0], device=self.device)
        self.gate_center = torch.tensor([6.0,0.0], device=self.device)
        self.last_action = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.last_action_no_scale = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        # Randomize robot friction
        env_ids = self.carter._ALL_INDICES
        mat_props = self.carter.root_physx_view.get_material_properties()
        mat_props[:, :, :2].uniform_(0.6, 0.8)
        self.carter.root_physx_view.set_material_properties(mat_props, env_ids.cpu())

    def _setup_scene(self):
        self.carter = Articulation(self.cfg.carter)
        self.ball = RigidObject(self.cfg.ball)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # add gate
        spawn_from_usd(prim_path="/World/envs/env_.*/gate", cfg=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Users/gate.usd"))
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["carter"] = self.carter
        self.scene.rigid_objects["ball"] = self.ball
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        # add diff controller
        # self._actions_joints[0] = ((2 * actions[0]) - (actions[1] * self.wheel_base)) / (2 * self.wheel_radius) * self.action_scale
        # self._actions_joints[1] = ((2 * actions[0]) + (actions[1] * self.wheel_base)) / (2 * self.wheel_radius) * self.action_scale
        self._processed_actions = self._actions * self.action_scale
        # print("self._processed_actions", self._processed_actions)

    def _apply_action(self) -> None:
        self.carter.set_joint_effort_target(self._processed_actions, joint_ids=[self._left_dof_idx[0], self._right_dof_idx[0]])
        # print([self._left_dof_idx[0], self._right_dof_idx[0]])

    def _get_observations(self) -> dict:
        # carter
        self.base_lin_vel = self.carter.data.root_lin_vel_w
        self.base_ang_vel = self.carter.data.root_ang_vel_w
        self.base_quat = self.carter.data.root_quat_w
        self.base_pos = self.carter.data.root_pos_w
        self.root_lin_vel_b = self.carter.data.root_lin_vel_b
        self.root_ang_vel_b = self.carter.data.root_ang_vel_b
        self.joint_vel = self.carter.data.joint_vel
        # ball
        self.ball_pos = self.ball.data.root_pos_w
        self.ball_lin_vel = self.ball.data.root_lin_vel_w

        carter_pos = self.base_pos - self.scene.env_origins # 3
        carter_lin_vel = self.root_lin_vel_b # 3
        carter_ang_vel = self.root_ang_vel_b # 3
        ball_pos = self.ball_pos - self.scene.env_origins
        self._previous_actions = self._actions.clone()

        obs = torch.cat(
            (   # 
                carter_pos, # 3
                carter_lin_vel, # 3
                carter_ang_vel, # 3
                self.carter.data.projected_gravity_b, # 3
                self.base_quat, # 4
                self.base_lin_vel, # 3
                self.base_ang_vel, # 3
                ball_pos, # 3
                self.ball_lin_vel, # 3
                self._commands, # 3
                self._actions, # 2
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        # carter
        self.base_lin_vel = self.carter.data.root_lin_vel_w
        self.base_ang_vel = self.carter.data.root_ang_vel_w
        self.base_quat = self.carter.data.root_quat_w
        self.base_pos = self.carter.data.root_pos_w
        carter_pos = self.base_pos - self.scene.env_origins
        self.root_lin_vel_b = self.carter.data.root_lin_vel_b
        self.root_ang_vel_b = self.carter.data.root_ang_vel_b
        self.joint_vel = self.carter.data.joint_vel
        # ball
        self.ball_pos = self.ball.data.root_pos_w
        ball_pos = self.ball_pos - self.scene.env_origins
        self.ball_lin_vel = self.ball.data.root_lin_vel_w
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, [0]] - self.carter.data.root_lin_vel_b[:, [0]]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self.carter.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # heading tracking
        # _, _, _, _, _, angle_to_target = compute_rot(
        # self.base_quat, self.base_lin_vel, self.base_ang_vel, self.ball_pos, self.base_pos)
        # heading_proj = torch.cos(angle_to_target)
        # heading_weight_tensor = torch.ones_like(heading_proj)
        # heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor,  heading_proj)
        carter_eular = math_utils.euler_xyz_from_quat(self.base_quat)
        carter_eular = torch.stack(carter_eular, dim=-1)
        carter_eular = math_utils.wrap_to_pi(carter_eular)
        target_dir = self.ball_pos - self.base_pos
        target_theta = torch.atan2(target_dir[:, 1], target_dir[:, 0])
        angle_to_target = target_theta - carter_eular[:, 2]
        angle_to_target = math_utils.wrap_to_pi(angle_to_target)
        heading_error = torch.sum(torch.square(angle_to_target.unsqueeze(1)), dim=1)
        heading_reward = torch.exp(-heading_error / 0.25)

        # position tracking
        distance = torch.sum(torch.square(ball_pos[:, :2] - carter_pos[:, :2]), dim=1)
        # distance_reward = (1. /(1. + torch.square(distance / self.cfg.position_target_sigma_soft)))
        distance_reward = torch.exp(-distance / 0.25)

        # out_of_bounds reward
        out_of_bounds = torch.any(torch.abs(ball_pos[:, [0]]) > self.cfg.max_ball_pos[0], dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(ball_pos[:,[1]]) > self.cfg.max_ball_pos[1], dim=1)
        out_of_bounds_reward = torch.where(out_of_bounds, 1.0, 0.0)

        # ball dis reward
        ball_dis = torch.sum(torch.square(ball_pos[:, :2] - self.gate_center), dim=1)
        ball_dis_reward = torch.exp(- ball_dis / 0.25)
        # print("ball_pos", self.ball_pos)
        # print("gate_center", self.gate_center)
        # print("ball_dis_reward", ball_dis_reward)

        # ball direction reward
        ball_vel_dir_theta = torch.atan2(self.ball_lin_vel[:, 1], self.ball_lin_vel[:, 0])
        ball_gate_theta = torch.atan2(self.gate_center[1] - self.ball_pos[:, 1], self.gate_center[0] - self.ball_pos[:, 0])
        ball_dir_diff = math_utils.wrap_to_pi(ball_vel_dir_theta - ball_gate_theta)
        ball_heading_error = torch.sum(torch.square(ball_dir_diff.unsqueeze(1)), dim=1)
        ball_heading_reward = torch.exp(-ball_heading_error / 0.25)
        

        # ball in gate reward
        ball_in_gate = torch.logical_and(
                torch.logical_and(
                    ball_pos[:, 0] > self.gate_x[0],
                    ball_pos[:, 0] < self.gate_x[1]
                ),
                torch.logical_and(
                    ball_pos[:, 1] > self.gate_y[0],
                    ball_pos[:, 1] < self.gate_y[1]
                )
            )

        ball_in_gate_reward = torch.sum(torch.where(ball_in_gate, 1.0, 0.0).unsqueeze(1), dim=1)


        # z velocity tracking
        z_vel_error = torch.square(self.carter.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self.carter.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self.carter.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self.carter.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self.carter.data.projected_gravity_b[:, :2]), dim=1)
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            # "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "track_heading": heading_reward * self.cfg.heading_reward_scale * self.step_dt,
            "track_distance": distance_reward * self.cfg.distance_reward_scale * self.step_dt,
            "out_of_bounds": out_of_bounds_reward * self.cfg.out_of_bounds_reward_scale * self.step_dt,
            "ball_dis": ball_dis_reward * self.cfg.ball_dis_reward_scale * self.step_dt,
            "ball_heading": ball_heading_reward * self.cfg.ball_heading_reward_scale * self.step_dt,
            "ball_in_gate": ball_in_gate_reward * self.cfg.ball_in_gate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            # "action_diff_l2": action_diff * self.cfg.action_rate_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        # print("lin_vel_error_mapped", lin_vel_error_mapped)
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        carter_eular = math_utils.euler_xyz_from_quat(self.base_quat)
        carter_eular = torch.stack(carter_eular, dim=-1)
        carter_eular = math_utils.wrap_to_pi(carter_eular)

        ball_pos = self.ball.data.root_state_w[:, :3] - self.scene.env_origins
        # out_of_bounds = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        out_of_bounds = torch.any(torch.abs(ball_pos[:, [0]]) > self.cfg.max_ball_pos[0], dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(ball_pos[:,[1]]) > self.cfg.max_ball_pos[1], dim=1)
        out_of_bounds = out_of_bounds | torch.any((torch.abs(carter_eular[:,[0]]) + torch.abs(carter_eular[:,[1]])) % (2 * torch.pi) > 0.8, dim=1)
        ball_in_gate = torch.where(
            torch.logical_and(
                torch.logical_and(
                    ball_pos[:, 0] > self.gate_x[0],
                    ball_pos[:, 0] < self.gate_x[1]
                ),
                torch.logical_and(
                    ball_pos[:, 1] > self.gate_y[0],
                    ball_pos[:, 1] < self.gate_y[1]
                )
            ),
            True,
            False
        )
        terminate = ball_in_gate | out_of_bounds
        # print((torch.abs(carter_eular[:,0]) + torch.abs(carter_eular[:,1])) % (2 * torch.pi))
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminate, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.carter._ALL_INDICES
        super()._reset_idx(env_ids)
        self.carter.reset(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids])
        self._commands[env_ids, 0] = sample_uniform(0.8, 1.2, (len(env_ids),), self.device)
        self._commands[env_ids, 2] = sample_uniform(-0.5, 0.5, (len(env_ids),), self.device)
        # print("self._commands", self._commands)
        # Reset robot state
        joint_pos = self.carter.data.default_joint_pos[env_ids]
        joint_vel = self.carter.data.default_joint_vel[env_ids]
        default_root_state = self.carter.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 0] += sample_uniform(
            - self.cfg.initial_carter_range[0],
            - 5.0,
            default_root_state[:, 0].shape,
            self.device,
        )
        default_root_state[:, 1] += sample_uniform(
            - 2.0,
            self.cfg.initial_carter_range[1],
            default_root_state[:, 1].shape,
            self.device,
        )
        self.carter.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.carter.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.carter.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Reset ball state
        ball_state = self.ball.data.default_root_state[env_ids]
        ball_state[:, :3] += self.scene.env_origins[env_ids]
        ball_state[:, 0] += sample_uniform(
            - self.cfg.initial_ball_range[0],
            self.cfg.initial_ball_range[0],
            ball_state[:, 0].shape,
            self.device,
        )
        ball_state[:, 1] += sample_uniform(
            - self.cfg.initial_ball_range[1],
            self.cfg.initial_ball_range[1],
            ball_state[:, 1].shape,
            self.device,
        )
        self.ball.write_root_state_to_sim(ball_state, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    

    
    # def _reward_stop(self):
    #     # Penalize stop
    #     return torch.sum(torch.square(self.carter.data.root_state_w[:, 7:9]), dim=1)
    
    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_action - self.actions), dim=1)
    
    # def _reward_out_of_bounds(self):
    #     ball_pos = self.ball.data.root_state_w[:, :3] - self.scene.env_origins
    #     return torch.any(torch.abs(ball_pos[:,0]) > self.cfg.max_ball_pos[0]) | torch.any(torch.abs(ball_pos[:,1]) > self.cfg.max_ball_pos[1])
    
    # def _reward_velo_dir(self):
    #     carter_vel = self.carter.data.root_state_w[:, 7:10]
    #     carter_eular = math_utils.euler_xyz_from_quat(self.carter.data.root_state_w[:, 3:7])
    #     carter_eular = torch.stack(carter_eular, dim=-1)
    #     carter_eular = torch.where(carter_eular > torch.pi, carter_eular - 2 * torch.pi, carter_eular)
    #     xy_diff = self.ball_pos - self.base_pos
    #     xy_diff = xy_diff / (0.001 + torch.norm(xy_diff, dim=1).unsqueeze(1))
    #     bad_dir = carter_vel[:,0] * xy_diff[:,0] + carter_vel[:,1] * xy_diff[:,1] < -0.25
    #     # good_dir = carter_vel[:,0] * xy_diff[:,0] + carter_vel[:,1] * xy_diff[:,1] > 0.25
    #     theta_tar = torch.atan2(xy_diff[:,1], xy_diff[:,0])
    #     theta_carter_vel = torch.atan2(carter_vel[:,1], carter_vel[:,0])
    #     diff1 = torch.abs(theta_tar - theta_carter_vel)
    #     diff2 = torch.abs(carter_eular[:, 2] - theta_tar)
    #     distance = torch.norm(self.ball.data.root_state_w[:, :2] - self.carter.data.root_state_w[:, :2], dim=1)
    #     reward =  torch.exp(-diff2) - 0.3
        
    #     # return bad_dir * 1.0 * (distance > self.cfg.position_target_sigma_soft)
    #     return reward
    

@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def compute_rewards(
    rew_scale_ang_vel_xy: float,
    rew_scale_stop: float,
    rew_scale_out_of_bounds: float,
    rew_scale_action_rate: float,
    rew_scale_vel_dir: float,
    rew_scale_speed: float,
    _reward_ang_vel_xy: torch.Tensor,
    _reward_stop: torch.Tensor,
    _reward_action_rate: torch.Tensor,
    _reward_out_of_bounds: torch.Tensor,
    _reward_vel_dir: torch.Tensor,
    _reward_speed: torch.Tensor,
):
    rew_ang_vel_xy = rew_scale_ang_vel_xy * _reward_ang_vel_xy
    rew_stop = rew_scale_stop * _reward_stop
    rew_action_rate = rew_scale_action_rate * _reward_action_rate
    rew_out_of_bounds = rew_scale_out_of_bounds * _reward_out_of_bounds
    rew_vel_dir = rew_scale_vel_dir * _reward_vel_dir
    rew_speed = rew_scale_speed * _reward_speed
    total_reward = rew_ang_vel_xy + rew_stop + rew_action_rate + rew_out_of_bounds + rew_vel_dir + rew_speed
    return total_reward


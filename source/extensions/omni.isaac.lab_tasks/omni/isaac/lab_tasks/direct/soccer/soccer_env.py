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
@configclass
class SoccerEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    action_scale = 50.0  # [N]
    action_bias = 0.0
    num_actions = 2
    num_observations = 25
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

    # gate : AssetBaseCfg (
    #     prim_path="/World/envs/env_.*/gate",
    #     spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Users/gate.usd",)
    # )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=22.0, replicate_physics=True)

    # reset
    max_ball_pos = [11.0, 7.0]  # the cart is reset if it exceeds that position [m]
    initial_carter_range = [4.0, 5.0]
    initial_ball_range = [3.0, 5.0]
    # reward scales
    rew_scale_ang_vel_xy = 0.0
    rew_scale_stop = -0.0
    rew_scale_out_of_bounds = 0.0
    rew_scale_action_rate = 0.0
    rew_scale_vel_dir = 1.0
    rew_scale_speed = 0.0
    position_target_sigma_soft = 2.0

class SoccerEnv(DirectRLEnv):
    cfg: SoccerEnvCfg

    def __init__(self, cfg: SoccerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._left_dof_idx, _ = self.carter.find_joints(self.cfg.left_dof_name)
        self._right_dof_idx, _ = self.carter.find_joints(self.cfg.right_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_vel = self.carter.data.joint_vel
        self.carter_state = self.carter.data.root_state_w
        self.gate_x = [10.0, 10.8]
        self.gate_y = [-0.9, 0.9]
        self.gate_center = torch.tensor([10.0,0.0], device=self.device)
        self.ball_state = self.ball.data.root_state_w
        self.base_lin_vel = self.carter.data.root_state_w[:, 7:10]
        self.last_action = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.last_action_no_scale = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

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
        self.last_action_no_scale = actions.clone()
        self.actions = self.action_scale * actions.clone()
        print("actions", self.actions)
        self.last_action = self.actions.clone()

    def _apply_action(self) -> None:
        self.carter.set_joint_velocity_target(self.actions, joint_ids=[self._left_dof_idx[0], self._right_dof_idx[0]])

    def _get_observations(self) -> dict:
        carter_pos = self.carter.data.root_state_w[:, :3] - self.scene.env_origins
        carter_eular = math_utils.euler_xyz_from_quat(self.carter.data.root_state_w[:, 3:7])
        carter_eular = torch.stack(carter_eular, dim=-1)
        carter_vel = self.carter.data.root_state_w[:, 7:10]
        vel_yaw = quat_rotate_inverse(yaw_quat(self.carter.data.root_quat_w), self.carter.data.root_lin_vel_w[:, :3])
        carter_ang_vel = self.carter.data.root_state_w[:, 10:13]
        ball_pos = self.ball.data.root_state_w[:, :3] - self.scene.env_origins
        ball_pos_carter = quat_rotate_inverse(self.carter.data.root_quat_w, ball_pos)
        ball_vel = self.ball.data.root_state_w[:, 7:10]
        last_action = self.last_action_no_scale

        obs = torch.cat(
            (   # 3 3 3 3 3 3 2 = 20 + 2 + 3
                carter_pos,
                carter_eular,
                carter_vel,
                vel_yaw,
                carter_ang_vel,
                ball_pos,
                ball_pos_carter,
                ball_vel,
                last_action,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # carter_pos = self.carter.data.root_state_w[:, :3] - self.scene.env_origins
        # carter_eular = math_utils.euler_xyz_from_quat(self.carter.data.root_state_w[:, 3:7])
        # carter_eular = torch.stack(carter_eular, dim=-1)
        # carter_vel = self.carter.data.root_state_w[:, 7:10]
        # carter_ang_vel = self.carter.data.root_state_w[:, 10:13]
        # ball_pos = self.ball.data.root_state_w[:, :3] - self.scene.env_origins
        # ball_vel = self.ball.data.root_state_w[:, 7:10]


        ang_vel_xy = self._reward_ang_vel_xy()
        stop = self._reward_stop()
        action_rate = self._reward_action_rate()
        out_of_bounds = self._reward_out_of_bounds()
        total_reward = compute_rewards(
            self.cfg.rew_scale_ang_vel_xy,
            self.cfg.rew_scale_stop,
            self.cfg.rew_scale_out_of_bounds,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_vel_dir,
            self.cfg.rew_scale_speed,
            ang_vel_xy,
            stop,
            action_rate,
            out_of_bounds,
            self._reward_velo_dir(),
            self._reward_speed(),
        )
        
        # total_reward = compute_rewards(
        #     self.cfg.rew_scale_alive,
        #     self.cfg.rew_scale_terminated,
        #     self.cfg.rew_scale_pole_pos,
        #     self.cfg.rew_scale_cart_vel,
        #     self.cfg.rew_scale_pole_vel,
        #     self.joint_pos[:, self._pole_dof_idx[0]],
        #     self.joint_vel[:, self._pole_dof_idx[0]],
        #     self.joint_pos[:, self._cart_dof_idx[0]],
        #     self.joint_vel[:, self._cart_dof_idx[0]],
        #     self.reset_terminated,
        # )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        carter_eular = math_utils.euler_xyz_from_quat(self.carter.data.root_state_w[:, 3:7])
        carter_eular = torch.stack(carter_eular, dim=-1)
        carter_eular = torch.where(carter_eular > torch.pi, carter_eular - 2 * torch.pi, carter_eular)
        ball_pos = self.ball.data.root_state_w[:, :3] - self.scene.env_origins
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        out_of_bounds = torch.any(torch.abs(ball_pos[:,0]) > self.cfg.max_ball_pos[0], dim=-1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(ball_pos[:,1]) > self.cfg.max_ball_pos[1], dim=-1)
        # out_of_bounds = out_of_bounds | torch.any((torch.abs(carter_eular[:,0]) + torch.abs(carter_eular[:,1])) % (2 * torch.pi) > 1.0, dim=-1)
        # print((torch.abs(carter_eular[:,0]) + torch.abs(carter_eular[:,1])) % (2 * torch.pi))
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.carter._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.carter.data.default_root_state[env_ids]
        default_root_state[:, 0] += sample_uniform(
            - self.cfg.initial_carter_range[0],
            self.cfg.initial_carter_range[0],
            default_root_state[:, 0].shape,
            self.device,
        )
        default_root_state[:, 1] += sample_uniform(
            - self.cfg.initial_carter_range[1],
            self.cfg.initial_carter_range[1],
            default_root_state[:, 1].shape,
            self.device,
        )
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        ball_state = self.ball.data.default_root_state[env_ids]
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
        ball_state[:, :3] += self.scene.env_origins[env_ids]

        self.carter.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.carter.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.carter.write_joint_state_to_sim(self.carter.data.default_joint_pos[env_ids], self.carter.data.default_joint_vel[env_ids], None, env_ids)
        self.ball.write_root_pose_to_sim(ball_state[:, :7], env_ids)
        self.ball.write_root_velocity_to_sim(ball_state[:, 7:], env_ids)
    
    def _reward_joint_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.carter.data.joint_vel[:, :2]), dim=1)
    
    def _reward_stop(self):
        # Penalize stop
        return torch.sum(torch.square(self.carter.data.root_state_w[:, 7:9]), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_action - self.actions), dim=1)
    
    def _reward_out_of_bounds(self):
        ball_pos = self.ball.data.root_state_w[:, :3] - self.scene.env_origins
        return torch.any(torch.abs(ball_pos[:,0]) > self.cfg.max_ball_pos[0]) | torch.any(torch.abs(ball_pos[:,1]) > self.cfg.max_ball_pos[1])
    
    def _reward_velo_dir(self):
        carter_vel = self.carter.data.root_state_w[:, 7:10]
        carter_eular = math_utils.euler_xyz_from_quat(self.carter.data.root_state_w[:, 3:7])
        carter_eular = torch.stack(carter_eular, dim=-1)
        carter_eular = torch.where(carter_eular > torch.pi, carter_eular - 2 * torch.pi, carter_eular)
        xy_diff = self.ball.data.root_state_w[:, :2] - self.carter.data.root_state_w[:, :2] 
        xy_diff = xy_diff / (0.001 + torch.norm(xy_diff, dim=1).unsqueeze(1))
        bad_dir = carter_vel[:,0] * xy_diff[:,0] + carter_vel[:,1] * xy_diff[:,1] < -0.25

        theta_tar = torch.atan2(xy_diff[:,1], xy_diff[:,0])
        theta_carter_vel = torch.atan2(carter_vel[:,1], carter_vel[:,0])
        diff1 = torch.abs(theta_tar - theta_carter_vel)
        diff2 = torch.abs(carter_eular[:, 2] - theta_tar)
        # print("diff1", diff1)
        # print("carter_eular", carter_eular)
        # print("theta_tar", theta_tar)
        # print("diff2", diff2)
        # distance = torch.norm(self.ball.data.root_state_w[:, :2] - self.carter.data.root_state_w[:, :2], dim=1)
        reward =  torch.exp(-diff2) - 0.3
        return reward
        # return bad_dir * 1.0 * (distance > self.cfg.position_target_sigma_soft)
    
    def _reward_speed(self):
        carter_vel = self.carter.data.root_state_w[:, 7:10]
        # carter_base_vel = quat_apply(self.carter.data.root_state_w[:, 3:7], carter_vel)
        vel_yaw = quat_rotate_inverse(yaw_quat(self.carter.data.root_quat_w), self.carter.data.root_lin_vel_w[:, :3])
        print("vel_yaw", vel_yaw)
        lin_vel_error = torch.sum(torch.square(1.0 - vel_yaw[:, 0]))
        # lin_vel_error = torch.sum(torch.square(1.0 - carter_vel[:, 0]))
        print("lin_vel_error", lin_vel_error)
        std = 1.0
        return torch.exp(-lin_vel_error)
        # print("carter_base_vel", carter_base_vel)
        carter_vel_norm = torch.norm(carter_vel, dim=1)
        # vel_reward = carter_base_vel[:,0] * torch.exp(1.0 - carter_vel_norm)
        vel_reward = 0
        
        # distance = torch.norm(self.ball.data.root_state_w[:, :2] - self.carter.data.root_state_w[:, :2], dim=1)
        # return (1. /(1. + torch.square(distance / self.cfg.position_target_sigma_soft)))


    # def _reward_

    # def _reward_nomove(self):
    #     # travel_dist = torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1)
    #     static = torch.logical_and(torch.norm(self.base_lin_vel[:,:2], dim=-1) < 0.1, torch.abs(self.base_ang_vel[:,2]) < 0.1)
    #     forward = quat_apply(self.base_quat, self.forward_vec)
    #     xy_dif = self.position_targets[:,:2] - self.root_states[:, :2]
    #     xy_dif = xy_dif / (0.001 + torch.norm(xy_dif, dim=1).unsqueeze(1))
    #     bad_dir = forward[:,0] * xy_dif[:,0] + forward[:,1] * xy_dif[:,1] < -0.25  # base orientation not -> target
    #     distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
    #     return static * bad_dir * 1.0 * (distance > self.cfg.rewards.position_target_sigma_soft)

    # def _reward_velo_dir(self):
    #     forward = quat_apply(self.base_quat, self.forward_vec)
    #     xy_dif = self.position_targets[:,:2] - self.root_states[:, :2]
    #     xy_dif = xy_dif / (0.001 + torch.norm(xy_dif, dim=1).unsqueeze(1))
    #     good_dir = forward[:,0] * xy_dif[:,0] + forward[:,1] * xy_dif[:,1] > -0.25  # base orientation -> target
    #     distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
    #     _rew = self.base_lin_vel[:,0].clip(min=0.0) * good_dir * (distance>self.cfg.rewards.position_target_sigma_tight) / 4.5 \
    #                                         + 1.0 * (distance<self.cfg.rewards.position_target_sigma_tight)
    #     return _rew

    # def _reward_reach_pos_target_soft(self):
    #     distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
    #     return (1. /(1. + torch.square(distance / self.cfg.rewards.position_target_sigma_soft))) * self._command_duration_mask(self.cfg.rewards.rew_duration)


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
    # rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    # rew_termination = rew_scale_terminated * reset_terminated.float()
    # rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    # rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    # total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    rew_ang_vel_xy = rew_scale_ang_vel_xy * _reward_ang_vel_xy
    rew_stop = rew_scale_stop * _reward_stop
    rew_action_rate = rew_scale_action_rate * _reward_action_rate
    rew_out_of_bounds = rew_scale_out_of_bounds * _reward_out_of_bounds
    rew_vel_dir = rew_scale_vel_dir * _reward_vel_dir
    rew_speed = rew_scale_speed * _reward_speed
    total_reward = rew_ang_vel_xy + rew_stop + rew_action_rate + rew_out_of_bounds + rew_vel_dir + rew_speed
    # print("rew_ang_vel_xy", rew_ang_vel_xy)
    # print("rew_acc", rew_stop)
    # print("rew_action_rate", rew_action_rate)
    # print("rew_out_of_bounds", rew_out_of_bounds)
    # print("rew_vel_dir", rew_vel_dir)
    # print("rew_speed", rew_speed)
    return total_reward


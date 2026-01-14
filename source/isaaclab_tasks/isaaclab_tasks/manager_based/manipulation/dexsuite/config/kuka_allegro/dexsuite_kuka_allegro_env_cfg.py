# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroReorientRewardCfg(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )


@configclass
class KukaAllegroMixinCfg:
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        finger_tip_body_list = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])


@configclass
class DexsuiteKukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroReorientEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # [修改] 降低目标高度范围，实现“只抓取/微抬”
        # 原始范围是 (0.55, 0.95)，桌面高度约 0.25，物体初始高度约 0.35
        # 改为 (0.38, 0.45) 意味着只需将物体提离桌面 10cm 左右即可
        # self.commands.object_pose.ranges.pos_z = (0.38, 0.45)

        # [可选] 如果想关闭红色点点可视化，取消下面这行的注释
        # self.observations.perception.object_point_cloud.params["visualize"] = False

@configclass
class DexsuiteKukaAllegroLiftEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass

# [New] Define the custom 3-finger robot configuration
# You must update the usd_path to point to your modified USD file
# MY_3FINGER_ROBOT_CFG = KUKA_ALLEGRO_CFG.copy()
# TODO: Update this path to your actual modified USD file
# MY_3FINGER_ROBOT_CFG.spawn.usd_path = "/workspace/isaaclab/source/usd/kuka/kuka.usd"

@configclass
class KukaAllegroThreeFingerRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroMixinCfg:
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        finger_tip_body_list = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])


@configclass
class KukaAllegroThreeFingerMixinCfg:
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroThreeFingerRelJointPosActionCfg = KukaAllegroThreeFingerRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        # Use the custom robot config with the modified USD
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Define the 3 fingers we want to use
        finger_tip_body_list = ["index_link_3", "middle_link_3", "thumb_link_3"]

        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        # Update observations to only use these 3 fingers
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),
        )

        # Note: Ensure your modified USD still has these tip bodies or adjust names accordingly.
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])

@configclass
class DexsuiteKukaAllegroThreeFingerLiftEnvCfg(KukaAllegroThreeFingerMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    # def __post_init__(self):
    #     super().__post_init__()
    #     # Apply the same lift task customizations
    #     self.commands.object_pose.ranges.pos_z = (0.55, 0.85)
    pass


@configclass
class DexsuiteKukaAllegroThreeFingerLiftEnvCfg_PLAY(KukaAllegroThreeFingerMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # [Explicit Control] Set the target lift height range for inference/play
        # Default table height ~0.25m.
        # Range (0.8, 0.85) means lifting the object 30-60cm above the table.
        self.commands.object_pose.ranges.pos_z = (0.8, 0.85)

        # [Explicit Control] Object Types for PLAY/Inference
        # Provide a list of different object configurations. 
        # The environment acts as a collection of these variants and selects one randomly on reset.
        from isaaclab.sim import SphereCfg, CuboidCfg, CapsuleCfg, RigidBodyMaterialCfg, ConeCfg
        
        mat_cfg = RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5)

        self.scene.object.spawn.assets_cfg = [
            #  CuboidCfg(size=(0.05, 0.05, 0.05), physics_material=mat_cfg),
            #  SphereCfg(radius=0.035, physics_material=mat_cfg),
            #  CapsuleCfg(radius=0.03, height=0.06, axis="Z", physics_material=mat_cfg),
             ConeCfg(radius=0.025, height=0.1, physics_material=mat_cfg),
        ]
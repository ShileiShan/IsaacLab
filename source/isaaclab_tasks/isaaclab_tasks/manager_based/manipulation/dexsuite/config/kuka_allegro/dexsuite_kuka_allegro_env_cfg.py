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
from isaaclab_assets.robots import KUKA_SELF_THREE_FINGER_CFG
from isaaclab_assets.robots import KUKA_7DOF_THREE_FINGER_CFG
from isaaclab_assets.robots import KUKA_7DOF_THREE_FINGER_CFG1
from isaaclab_assets.robots import KUKA_7DOF_THREE_FINGER_RIGHT_ARM_CFG

from ... import dexsuite_env_cfg as dexsuite
from ... import dexsuite_right_arm_env_cfg as dexsuite_right_arm
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
    # [FIX] Increase scale to allow larger movements per step.
    # Was 0.1, trying 0.5 to see if fingers move more visibly and can reach the object.
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)
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


@configclass
class KukaAllegroReorientRewardCfg2(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )

@configclass
class KukaAllegroThreeFingerRelJointPosActionCfg2:
    # [FIX] Increase scale to allow larger movements per step.
    # Was 0.1, trying 0.5 to see if fingers move more visibly and can reach the object.
    # [FIX] Filter out 'finger_joint1_1' by using a regex that excludes it.
    # The regex below matches all joints except specifically 'finger_joint1_1'.
    # It allows: iiwa joints, and finger joints 1_2, 1_3, 1_4, plus all 2_x and 3_x.
    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", 
        joint_names=["iiwa.*", "finger_joint1_[2-4]", "finger_joint[2-3]_.*"], 
        scale=0.2
    )

@configclass
class KukaSelfThreeFingerMixinCfg:
    rewards: KukaAllegroReorientRewardCfg2 = KukaAllegroReorientRewardCfg2()
    actions: KukaAllegroThreeFingerRelJointPosActionCfg2 = KukaAllegroThreeFingerRelJointPosActionCfg2()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "base_link"
        # Use the custom robot config from kuka_allegro_three_finger.py
        self.scene.robot = KUKA_SELF_THREE_FINGER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Define the 3 fingers we want to use.
        # Note: These link names must match the ones in the USD file used by KUKA_SELF_THREE_FINGER_CFG.
        finger_tip_body_list = ["Empty_Link1_4", "Empty_Link2_3", "Empty_Link3_3"]

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

        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["base_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["base_link", ".*_tip"])


@configclass
class DexsuiteKukaSelfThreeFingerLiftEnvCfg(KukaSelfThreeFingerMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaSelfThreeFingerLiftEnvCfg_PLAY(KukaSelfThreeFingerMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.ranges.pos_z = (0.8, 0.85)
        from isaaclab.sim import SphereCfg, CuboidCfg, CapsuleCfg, RigidBodyMaterialCfg, ConeCfg
        # from isaaclab.sim import RigidBodyMaterialCfg, ConeCfg
        mat_cfg = RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5)
        self.scene.object.spawn.assets_cfg = [
             CapsuleCfg(radius=0.03, height=0.06, axis="Z", physics_material=mat_cfg),
             ConeCfg(radius=0.025, height=0.1, physics_material=mat_cfg),
        ]




@configclass
class Kuka7DOFReorientRewardCfg(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts_7DOF,
        weight=2.0,
        params={"threshold": 1.0},
    )

    thumb_open = RewTerm(
        func=mdp.reward_thumb_open,
        weight=0.1,
        params={
            "target": 2.0,  # [New] 目标张开角度
            "std": 0.5,  # [New] 根据需要调整这个参数
            "asset_cfg": SceneEntityCfg("robot", joint_names=["thumb_joint1"]),
        },
    )

    grasp_posture_reward = RewTerm(
        func=mdp.reward_grasp_status,
        weight=0.05, # 权重可调
        params={
            "std": 1.0, # 敏感度参数
            "asset_cfg": SceneEntityCfg("robot"),
            "targets": {
                "thumb_joint1": 2.5,
                "thumb_joint2": 0.0,
                "thumb_joint3": -0.0,
                "middle_joint1": 0.0,
                "middle_joint2": -0.0,
                "index_joint1": 0.0,
                "index_joint2": -0.0,
            },
        },
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh_7DOF,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    # hand_above_object = RewTerm(
    #     func=mdp.reward_hand_above_object,
    #     weight=0.5,
    #     params={
    #         "threshold": 0.0,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["grasp_center", "Empty_Link2", "Empty_Link3"]),
    #         "object_cfg": SceneEntityCfg("object"),
    #     },
    # )

@configclass
class Kuka7DOFThreeFingerRelJointPosActionCfg2:
    # [FIX] Increase scale to allow larger movements per step.
    # Was 0.1, trying 0.5 to see if fingers move more visibly and can reach the object.
    # [FIX] Filter out 'finger_joint1_1' by using a regex that excludes it.
    # The regex below matches all joints except specifically 'finger_joint1_1'.
    # It allows: iiwa joints, and finger joints 1_2, 1_3, 1_4, plus all 2_x and 3_x.
    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.1
    )

@configclass
class Kuka7DOFThreeFingerMixinCfg:
    rewards: Kuka7DOFReorientRewardCfg = Kuka7DOFReorientRewardCfg()
    actions: Kuka7DOFThreeFingerRelJointPosActionCfg2 = Kuka7DOFThreeFingerRelJointPosActionCfg2()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "grasp_center"
        # Use the custom robot config from kuka_allegro_three_finger.py
        self.scene.robot = KUKA_7DOF_THREE_FINGER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Define the 3 fingers we want to use.
        # Note: These link names must match the ones in the USD file used by KUKA_7DOF_THREE_FINGER_CFG.
        finger_tip_body_list = ["thumb_tip", "middle_tip", "index_tip", "Empty_Link1_2", "Empty_Link2", "Empty_Link3"]

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

        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["grasp_center", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["grasp_center", ".*_tip"])


######################################### cfg 1

@configclass
class Kuka7DOFReorientRewardCfg1(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts_7DOF1,
        weight=2.0,
        params={"threshold": 1.0},
    )

    fingers_to_object = RewTerm(func=mdp.object_ee_distance1, params={"std": 0.4}, weight=1.0)

    thumb_open = RewTerm(
        func=mdp.reward_thumb_open,
        weight=0.1,
        params={
            "target": 2.0,  # [New] 目标张开角度
            "std": 0.5,  # [New] 根据需要调整这个参数
            "asset_cfg": SceneEntityCfg("robot", joint_names=["thumb_joint1"]),
        },
    )

    grasp_posture_reward = RewTerm(
        func=mdp.reward_grasp_status,
        weight=0.05, # 权重可调
        params={
            "std": 1.0, # 敏感度参数
            "asset_cfg": SceneEntityCfg("robot"),
            "targets": {
                "thumb_joint1": 2.5,
                "thumb_joint2": 0.0,
                "thumb_joint3": -0.0,
                "middle_joint1": 0.0,
                "middle_joint2": -0.0,
                "index_joint1": 0.0,
                "index_joint2": -0.0,
            },
        },
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh_7DOF1,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class Kuka7DOFThreeFingerMixinCfg1:
    rewards: Kuka7DOFReorientRewardCfg1 = Kuka7DOFReorientRewardCfg1()
    actions: Kuka7DOFThreeFingerRelJointPosActionCfg2 = Kuka7DOFThreeFingerRelJointPosActionCfg2()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "grasp_center"
        # Use the custom robot config from kuka_allegro_three_finger.py
        self.scene.robot = KUKA_7DOF_THREE_FINGER_CFG1.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Define the 3 fingers we want to use.
        # Note: These link names must match the ones in the USD file used by KUKA_7DOF_THREE_FINGER_CFG.
        finger_tip_body_list = ["Empty_Link1_3", "Empty_Link2_1", "Empty_Link3_1", "Empty_Link1_2", "Empty_Link2", "Empty_Link3"]

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

        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["grasp_center", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["grasp_center", ".*_tip"])




@configclass
class DexsuiteKuka7DOFThreeFingerLiftEnvCfg1(Kuka7DOFThreeFingerMixinCfg1, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKuka7DOFThreeFingerLiftEnvCfg1_PLAY(Kuka7DOFThreeFingerMixinCfg1, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.ranges.pos_z = (0.8, 0.85)
        from isaaclab.sim import SphereCfg, CuboidCfg, CapsuleCfg, RigidBodyMaterialCfg, ConeCfg
        # from isaaclab.sim import RigidBodyMaterialCfg, ConeCfg
        mat_cfg = RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5)
        self.scene.object.spawn.assets_cfg = [
            #  CapsuleCfg(radius=0.02, height=0.06, axis="Z", physics_material=mat_cfg),
             ConeCfg(radius=0.02, height=0.1, physics_material=mat_cfg),
            #  CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            #  SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        ]

########################################### right arm
@configclass
class Kuka7DOFRightArmReorientRewardCfg(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contact_bonus_scored,
        weight=2.0,
        params={"threshold": 1.0},
    )

    fingers_to_object = RewTerm(func=mdp.object_ee_distance2, params={"std": 0.3}, weight=1.0)

    thumb_open = RewTerm(
        func=mdp.reward_thumb_open,
        weight=0.1,
        params={
            "target": 2.0,  # [New] 目标张开角度
            "std": 0.5,  # [New] 根据需要调整这个参数
            "asset_cfg": SceneEntityCfg("robot", joint_names=["thumb_joint1"]),
        },
    )

    # grasp_posture_reward = RewTerm(
    #     func=mdp.reward_grasp_status,
    #     weight=0.1, # 权重可调
    #     params={
    #         "std": 0.4, # 敏感度参数
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "targets": {
    #             "thumb_joint1": 2.5,
    #             "thumb_joint2": 0.0,
    #             "thumb_joint3": -0.0,
    #             "middle_joint1": 0.0,
    #             "middle_joint2": -0.0,
    #             "index_joint1": 0.0,
    #             "index_joint2": -0.0,
    #         },
    #     },
    # )

    

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh_bouns,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    hand_above_object = RewTerm(
        func=mdp.reward_hand_above_object,
        weight=0.05,
        params={
            "threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", body_names=["grasp_center", "Empty_Link2", "Empty_Link3"]),
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    splay_penalty = RewTerm(
        func=mdp.penalty_tip_status,
        weight=-0.5,
        params={
            "threshold": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class Kuka7DOFThreeFingerRightArmMixinCfg:
    rewards: Kuka7DOFRightArmReorientRewardCfg = Kuka7DOFRightArmReorientRewardCfg()
    actions: Kuka7DOFThreeFingerRelJointPosActionCfg2 = Kuka7DOFThreeFingerRelJointPosActionCfg2()

    def __post_init__(self: dexsuite_right_arm.DexsuiteReorientEnvCfg):
        super().__post_init__()
        # [FIX] Use strict regex to match the root base_link and avoid ambiguity with three_finger_right/base_link
        # self.commands.object_pose.body_name = "^base_link$"
        # self.commands.object_pose.ranges.pos_z = (1.3, 1.7)
        # self.commands.object_pose.ranges.pos_x = (0.4, 0.6)
        # self.commands.object_pose.ranges.pos_y = (-0.2, 0.2)
        # Use the custom robot config from kuka_allegro_three_finger.py
        self.scene.robot = KUKA_7DOF_THREE_FINGER_RIGHT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Define the 3 fingers we want to use.
        # Note: These link names must match the ones in the USD file used by KUKA_7DOF_THREE_FINGER_CFG.
        finger_tip_body_list = ["thumb_tip", "middle_tip", "index_tip", "Empty_Link1_2", "Empty_Link2", "Empty_Link3"]
        # finger_tip_body_list = ["Empty_Link1_3", "Empty_Link2_1", "Empty_Link3_1", "Empty_Link1_2", "Empty_Link2", "Empty_Link3"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/three_finger_right/" + link_name,
                    # filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    # 如果机器人指尖也有子层级（比如 collisions/mesh_0），建议也加 .*
                    # prim_path="{ENV_REGEX_NS}/Robot/three_finger_right/" + link_name + "/.*",
                    
                    # 2. 关键修正：使用 .* 匹配 Object 下的所有 Collider 子物体
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/.*"],                    
                ),
            )

        # Update observations to only use these 3 fingers
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),
        )

        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["grasp_center", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["grasp_center", ".*_tip"])

@configclass
class DexsuiteKuka7DOFThreeFingerRightArmLiftEnvCfg(Kuka7DOFThreeFingerRightArmMixinCfg, dexsuite_right_arm.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKuka7DOFThreeFingerRightArmLiftEnvCfg_PLAY(Kuka7DOFThreeFingerRightArmMixinCfg, dexsuite_right_arm.DexsuiteLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # [NEW] Add gino_robot exclusively for PLAY/Evaluation
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        import math

        self.commands.object_pose.ranges.pos_y=(-0.05, 0.15)
        self.commands.object_pose.ranges.pos_z=(0, 0.35)
        # self.commands.object_pose.debug_viz = False
        # [修改] 关闭物体点云的可视化
        # self.observations.perception.object_point_cloud.params["visualize"] = False
        
        # [修改] 提高视角高度
        # 原来的配置可能是 eye=(-2.25, 0.0, 1.75)，将 Z 值调大即可
        # 这里设置为 2.5，你可以根据需要调整
        self.viewer.eye = (-2.25, 0.0, 2.2)
        # self.viewer.lookat = (0.0, 0.0, 1.0)
        self.scene.table.init_state = RigidObjectCfg.InitialStateCfg(pos=(-0.65, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0))


        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]
        from isaaclab.sim import SphereCfg, CuboidCfg, CapsuleCfg, RigidBodyMaterialCfg, ConeCfg
        # from isaaclab.sim import RigidBodyMaterialCfg, ConeCfg
        import isaaclab.sim as sim_utils
        mat_cfg = RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5)
        # self.scene.object.spawn.assets_cfg = [
        #      CapsuleCfg(radius=0.02, height=0.06, physics_material=mat_cfg),
        #      ConeCfg(radius=0.03, height=0.1, physics_material=mat_cfg),
        #      CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        #      SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        # ]

        def read_json_config(json_path):
            import json
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                print(f"读取 JSON 失败: {e}")
                return None

        root_path = "/workspace/isaaclab/source/usd/assets_for_grasp/"
        config_path = "assets_for_grasp.json"
        targets = "url"
        data = read_json_config(root_path + config_path)
        urls = data[targets] if data and targets in data else []
        usd_path = root_path + urls[12] 

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 1.115)),
            # collision_group=-1,
        )

        # [New] Load multiple USD assets from index 11 to 14
        # import os
        # usd_assets_list = []
        # print(f"[INFO] Asset Config: Found {len(urls)} URLs. Checking indices 11-14.")
        
        # for i in range(11, 15):
        #     if i < len(urls):
        #         fname = urls[i]
        #         usd_path = root_path + fname
                
        #         # Check for existence
        #         if not os.path.exists(usd_path):
        #              print(f"[WARNING] Asset file missing at index {i}: {usd_path}")
        #              continue

        #         print(f"[INFO] Loading asset {i}: {fname}")
        #         usd_assets_list.append(
        #             sim_utils.UsdFileCfg(
        #                 usd_path=usd_path,
        #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #                     solver_position_iteration_count=16,
        #                     solver_velocity_iteration_count=0,
        #                     disable_gravity=False,
        #                 ),
        #                 collision_props=sim_utils.CollisionPropertiesCfg(),
        #                 mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        #             )
        #         )
        
        # if not usd_assets_list:
        #     print("[ERROR] No assets loaded! Using fallback.")
        #     from isaaclab.sim import CapsuleCfg
        #     usd_assets_list.append(CapsuleCfg(radius=0.03, height=0.06, axis="Z", physics_material=mat_cfg))
        # self.scene.object.spawn.assets_cfg = usd_assets_list
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=usd_assets_list),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 1.115)),
        #     # collision_group=-1,
        # )

        self.scene.gino_robot = AssetBaseCfg(
            prim_path="/World/envs/env_.*/gino_robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/workspace/isaaclab/source/usd/gino_robot0/gino_robot_visual.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0), 
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

"""
Script to calculate the degrees of freedom (DOF) of a robot using regex matching on joint names.
Usage: ./isaaclab.sh -p scripts/cal_joints.py
"""

import re
import omni.usd
from pxr import Usd, UsdPhysics





def main():
    # 1. 设置 USD 路径 (必须与训练配置中的完全一致)
    robot_path = "/kuka"
    # 2. 设置正则表达式 (必须与训练配置中的完全一致)
    # 你的 Training Config 中使用的是 joint_names=[".*"]
    regex_patterns = [".*"]
    
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(robot_path)
    # 3. 收集所有关节
    all_joint_paths = []
    
    print("-" * 50)
    print("SCANNING JOINTS...")
    
    # Traverse stage to find all joints
    for prim in stage.Traverse():
        # Check if it is a Revolute Joint (1 DOF) or Prismatic Joint (1 DOF)
        # Note: D6 Joints are more complex, but for Kuka/Allegro they are usually Revolute
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            # 获取关节名字，通常是 Prim 的最后一级名字
            joint_name = prim.GetName()
            all_joint_paths.append(joint_name)
            # print(f"Found Joint: {joint_name} ({prim.GetPath()})")

    # 4. 根据正则筛选
    matched_joints = []
    for joint_name in all_joint_paths:
        for pattern in regex_patterns:
            if re.fullmatch(pattern, joint_name):
                matched_joints.append(joint_name)
                break # Matched one pattern, move to next joint

    print("-" * 50)
    print(f"Total Joints Found in USD: {len(all_joint_paths)}")
    print(f"Joints Matched by Regex {regex_patterns}: {len(matched_joints)}")
    print("-" * 50)
    print("MATCHED JOINTS LIST:")
    for j in matched_joints:
        print(f"  - {j}")
    print("-" * 50)
    print(f"CALCULATED DOF: {len(matched_joints)}")
    print("-" * 50)

main()
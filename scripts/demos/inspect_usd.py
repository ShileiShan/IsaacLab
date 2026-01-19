# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to inspect the hierarchy of a USD file.
Usage:
    ./isaaclab.sh -p scripts/demos/inspect_usd.py
"""

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd, UsdGeom
import os

# Path to your USD file
USD_PATH = "/workspace/isaaclab/source/usd/assemed_robot/kuka/kuka_three_finger.usd"

def main():
    if not os.path.exists(USD_PATH):
        print(f"[jERROR] USD file not found: {USD_PATH}")
        return

    print(f"[INFO] Opening USD stage: {USD_PATH}")
    stage = Usd.Stage.Open(USD_PATH)
    
    print("\n" + "="*50)
    print("USD Hierarchy (Rigid Bodies / Links):")
    print("="*50)

    # Traverse all prims in the stage
    for prim in stage.Traverse():
        # Check if it's potentially a Link (Xform or RigidBody)
        # We filter for UsdGeom.Imageable to skip internal specific nodes, 
        # but showing everything is safer to find the path.
        path = str(prim.GetPath())
        name = prim.GetName()
        type_name = prim.GetTypeName()
        
        # Simple filter to make output readable (show Xform, Mesh, Capsule, etc)
        # Highlight our target links
        if "Empty_Link" in name or "ee_link" in name or "tip" in name:
            print(f" -> [TARGET?] {path} ({type_name})")
        else:
            # Print minimal info for context
            # Only print down to a certain depth or specific types if list is too long
            print(f"    {path} ({type_name})")

    print("\n" + "="*50)
    print("CHECK TASKS:")
    print("1. Find 'ee_link' in the list above.")
    print("2. Find 'Empty_Link...' in the list above.")
    print("3. Check if 'Empty_Link' is really a child of 'ee_link'.")
    print("   Example: /Robot/ee_link/Empty_Link1_4  <-- This structure is what your config expects.")
    print("   If it looks like /Robot/Empty_Link1_4, you need to remove 'ee_link/' from your config.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

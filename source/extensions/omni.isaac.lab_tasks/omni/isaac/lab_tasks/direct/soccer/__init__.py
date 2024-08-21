# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .soccer_env import SoccerEnv, SoccerEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Soccer-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.soccer:SoccerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SoccerEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.SoccerPPORunnerCfg,
    },
)
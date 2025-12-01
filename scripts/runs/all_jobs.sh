#!/bin/bash
set +e
# source this job's dir + job_vars.sh

source $(dirname $0)/job_vars.sh

launch sokoban "sokoban_coord_3b_base_ppo_think_s_entvarfilter" True ppo s 8 800 "${entvar_filter_overrides[@]}"
wait_sleep_reset

launch frozenlake "frozenlake_coord_3b_base_ppo_think_rolloutfilterratio0.75" True ppo void 8 800 "${filter_ratio_0_75_overrides[@]}"
wait_sleep_reset

launch sudoku "sudoku_4x4_3b_base_ppo_think_s" True ppo s 8 800
wait_sleep_reset

launch spatial "spatial_3b_base_ppo_think_s" True ppo s 8 800
wait_sleep_reset

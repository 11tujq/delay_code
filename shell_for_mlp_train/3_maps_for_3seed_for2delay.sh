#!/bin/bash
# This script trains 3 different Mujoco tasks with 3 random seeds and 2 delays.

set -euo pipefail

source /mlx_devbox/users/tujunqi/playground/yes/bin/activate delay-env
cd /mlx_devbox/users/tujunqi/playground/delay_code

export UDATADIR=/mlx_devbox/users/tujunqi/playground/delay_bytedance/delay_bytedance/data # directory for dataset
export UPRJDIR=/mlx_devbox/users/tujunqi/playground/delay_bytedance/delay_bytedance/code # directory for code
export UOUTDIR=/mlx_devbox/users/tujunqi/playground/delay_bytedance/delay_bytedance/output # directory for outputs such as logs
export NUM_WORKERS=0 # number of workers to use
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mlx_devbox/users/tujunqi/playground/.mujoco/mujoco210/bin

random_seeds=(3412 4143 5131)
delays=(5 10)
fixed_delay=false
maps=("HalfCheetah-v4" "Walker2d-v4" "Hopper-v4")

for delay in "${delays[@]}"; do
	for seed in "${random_seeds[@]}"; do
		for map in "${maps[@]}"; do
			echo "Training on map: $map with delay: $delay and seed: $seed"
			python src/entry.py \
				experiment=cat_mlp \
				env.name="$map" \
				env.fixed_delay="$fixed_delay" \
				env.delay="$delay" \
				seed="$seed" \
				task_name="train_delay_${delay}_seed_${seed}_map_${map}"
		done
	done
done

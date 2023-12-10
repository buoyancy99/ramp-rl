#!/bin/bash

# shellcheck disable=SC1090
source ~/.bashrc
module load anaconda/2022a
mkdir -p "/state/partition1/user/$USER"
cp -r "/home/gridsan/$USER/Libraries/mujoco-py" "/state/partition1/user/$USER"
export PYTHONPATH=.:"/state/partition1/user/$USER/mujoco-py"
cd ~/Projects/ramp-rl/
echo "build mujoco-py..."
python -c "import mujoco_py" || exit
echo "mujoco_py built successfully"

export DISABLE_TQDM=1

envArray=(
 "ReachWallEnv-v2"
 "DclawEnv-v1"
)


ENV_ID=${envArray[$LLSUB_RANK]}

echo "My task ID:  $LLSUB_RANK"
echo "Number of Tasks: $LLSUB_SIZE"
echo "Env ID:  $ENV_ID"

CUDA_VISIBLE_DEVICES=0,1 python experiments/train_expert.py --env_id "$ENV_ID" --total_steps 1000000 --threads 10

echo "Everything Done!"

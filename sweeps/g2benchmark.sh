#!/bin/bash

# shellcheck disable=SC1090
source ~/.bashrc
conda activate ramp-rl
cd ~/ramp-rl/ || exit # change here!
python -c "import mujoco_py" || exit
echo "mujoco_py built successfully"

export WANDB_MODE=online
export WANDB_START_METHOD=thread
export DISABLE_TQDM=1

export MY_WANDB_ID=your_wandb_id_here # change here!

envArray=(
  "ReachWallEnv-v2"
  "DclawEnv-v1"
)

epsArray=(
  0.5
  0.75
)

ENV_ID=${envArray[$LLSUB_RANK]}
ENV_EPS=${epsArray[$LLSUB_RANK]}

echo "My task ID:  $LLSUB_RANK"
echo "Number of Tasks: $LLSUB_SIZE"
echo "Env ID:  $ENV_ID"

FILE="ckpts/$ENV_ID/policy_0.zip"
if [ -f "$FILE" ]; then
    echo "policy file $FILE exists, use soft link."
else
  echo "policy file for state not found, train new policies..."
  CUDA_VISIBLE_DEVICES=0,1 python experiments/train_expert.py --env_id "$ENV_ID" --total_steps 500000 --threads 10
fi

if [ ! -f "ckpts/${ENV_ID}" ]; then
  cd ckpts
  ln -s "${ENV_STATE_ID}" "${ENV_ID}"
  cd ..
fi

CUDA_VISIBLE_DEVICES=0,1 python experiments/collect_rollouts.py --env_id "$ENV_ID" --basis_type rand --steps 32000 --eps "$ENV_EPS" --threads 10
cd "buffer/$ENV_ID"
ln -s "rand_${ENV_EPS}_2048" rand_2048
ln -s "rand_${ENV_EPS}_2048" "rand_${ENV_EPS}_256"
cd ../..
CUDA_VISIBLE_DEVICES=0 python experiments/collect_rollouts.py --env_id "$ENV_ID" --basis_type learned --basis_dim 256 --steps 32000 --eps "${epsArray[$LLSUB_RANK]}"
cd "buffer/$ENV_ID"
ln -s "learned_${ENV_EPS}_256" learned_256
rm "rand_${ENV_EPS}_256"
cd ../..

CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --algo mpc --env_id "$ENV_ID" --seed 0
CUDA_VISIBLE_DEVICES=1 python experiments/benchmark.py --algo mpc --env_id "$ENV_ID" --seed 1 &
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --algo mpc --env_id "$ENV_ID" --seed 2 &
CUDA_VISIBLE_DEVICES=1 python experiments/benchmark.py --algo mpc --env_id "$ENV_ID" --seed 3 &
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --algo mpc --env_id "$ENV_ID" --seed 4
wait
CUDA_VISIBLE_DEVICES=1 python experiments/benchmark.py --algo mpc --env_id "$ENV_ID" --seed 5 &
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --algo mpc --env_id "$ENV_ID" --seed 6 &
CUDA_VISIBLE_DEVICES=1 python experiments/benchmark.py --algo mpc --env_id "$ENV_ID" --seed 7 &
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --algo mpc --env_id "$ENV_ID" --seed 8
wait

CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --algo sf --basis_type learned --basis_dim 256 --env_id "$ENV_ID" --seed 0
CUDA_VISIBLE_DEVICES=1 python experiments/benchmark.py --algo sf --basis_type learned --basis_dim 256 --env_id "$ENV_ID" --seed 1 &
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --algo sf --basis_type learned --basis_dim 256 --env_id "$ENV_ID" --seed 2 &
CUDA_VISIBLE_DEVICES=1 python experiments/benchmark.py --algo sf --basis_type learned --basis_dim 256 --env_id "$ENV_ID" --seed 3 &
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --algo sf --basis_type learned --basis_dim 256 --env_id "$ENV_ID" --seed 4
wait
CUDA_VISIBLE_DEVICES=1 python experiments/benchmark.py --algo sf --basis_type learned --basis_dim 256 --env_id "$ENV_ID" --seed 5 &
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --algo sf --basis_type learned --basis_dim 256 --env_id "$ENV_ID" --seed 6 &
CUDA_VISIBLE_DEVICES=1 python experiments/benchmark.py --algo sf --basis_type learned --basis_dim 256 --env_id "$ENV_ID" --seed 7 &
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --algo sf --basis_type learned --basis_dim 256 --env_id "$ENV_ID" --seed 8
wait


echo "Everything Done!"

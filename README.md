# RaMP-RL
This is the code base for [
Self-Supervised Reinforcement Learning that Transfers using Random Features](https://boyuan.space/ramp-rl/).

```
@misc{chen2023selfsupervised,
      title={Self-Supervised Reinforcement Learning that Transfers using Random Features}, 
      author={Boyuan Chen and Chuning Zhu and Pulkit Agrawal and Kaiqing Zhang and Abhishek Gupta},
      year={2023},
      eprint={2305.17250},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Environment Setup
```angular2html
conda create -n ramp-rl python=3.8
conda activate ramp-rl
pip install -r requirement.txt
cd ramp-rl && export PYTHONPATH=.
export MY_WANDB_ID=your_wandb_id_here
```

Then, setup [wandb](https://docs.wandb.ai/quickstart) on your computer for logging.


## Generate data
Right now generating data is very slow, but we plan to release a set of pre-trained policy for your convenience soon


```angular2html
# Train expert policies for rollout collection
python experiments/train_expert.py --env_id ReachWallEnv-v2 --total_steps 500000 --threads 5

# Collect rollouts
python experiments/collect_rollouts.py --env_id ReachWallEnv-v2 --basis_type rand --steps 16000 --eps 0.8
```

## Training
```
# Run our method
python experiments/benchmark.py --algo mpc --env_id ReachWallEnv-v2 --seed 0

# Successor Feature
python experiments/benchmark.py --algo sf --basis_type learned --basis_dim 256 --env_id ReachWallEnv-v2 --seed 0
```

## End2end bash script 
We provide an end2end bash script in `sweeps/g2benchmark.sh`. Make sure you modify all the `# change here!` fields.
The script assumes you have 2 GPUs but changing it to 1 should be easy.

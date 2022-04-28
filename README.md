# Stochastic Weighted Twin Delayed Deep Deterministic Policy Gradient (SWTD3)
PyTorch implementation of the _Stochastic Weighted Twin Delayed Deep Deterministic Policy Gradient_ algorithm (SWTD3). 
Note that the implementation of the TD3 algorithm is heavily based on the [author's Pytorch implementation of the TD3 algorithm](https://github.com/sfujim/TD3). 

The algorithm is tested on [MuJoCo](https://gym.openai.com/envs/#mujoco) and [Box2D](https://gym.openai.com/envs/#box2d) continuous control tasks.

### Computing Infrastructure
Following computing infrastructure is used to produce the results.
| Hardware/Software  | Model/Version |
| ------------- | ------------- |
| Operating System  | Ubuntu 18.04.5 LTS  |
| CPU  | AMD Ryzen 7 3700X 8-Core Processor |
| GPU  | Nvidia GeForce RTX 2070 SUPER |
| CUDA  | 11.1  |
| Python  | 3.8.5 |
| PyTorch  | 1.8.1 |
| OpenAI Gym  | 0.17.3 |
| MuJoCo  | 1.50 |
| Box2D  | 2.3.10 |
| NumPy  | 1.19.4 |

### Usage
```
usage: main.py [-h] [--policy POLICY] [--env ENV] [--seed SEED] [--gpu GPU]
               [--start_time_steps N] [--buffer_size BUFFER_SIZE]
               [--eval_freq N] [--max_time_steps N] [--exploration_noise G]
               [--batch_size N] [--discount G] [--tau G] [--policy_noise G]
               [--noise_clip G] [--policy_freq N] [--save_model]
               [--load_model LOAD_MODEL]
```

### Arguments
```
optional arguments:
  -h, --help            show this help message and exit
  --policy POLICY       Algorithm (default: SWTD3)
  --env ENV             OpenAI Gym environment name
  --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym (default: 0)
  --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
  --start_time_steps N  Number of exploration time steps sampling random actions (default: 1000)
  --buffer_size BUFFER_SIZE Size of the experience replay buffer (default: 1000000)
  --eval_freq N         Evaluation period in number of time steps (default: 1000)
  --max_time_steps N    Maximum number of steps (default: 1000000)
  --exploration_noise G Std of Gaussian exploration noise
  --batch_size N        Batch size (default: 256)
  --discount G          Discount factor for reward (default: 0.99)
  --tau G               Learning rate in soft/hard updates of the target networks (default: 0.005)
  --policy_noise G      Noise added to target policy during critic update
  --noise_clip G        Range to clip target policy noise
  --policy_freq N       Frequency of delayed policy updates
  --save_model          Save model and optimizer parameters
  --load_model LOAD_MODEL Model load file name; if empty, does not load
  ```

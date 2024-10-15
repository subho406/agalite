# AGaLiTe: Approximate Gated Linear Transformers for Online Reinforcement Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

***(Published in Transactions on Machine Learning Research)***

**Paper URL:** https://openreview.net/forum?id=lh6vOAHuvo

**Abstract:** In this paper we investigate transformer architectures designed for partially observable online reinforcement learning. The self-attention mechanism in the transformer architecture is capable of capturing long-range dependencies and it is the main reason behind its effectiveness in processing sequential data. Nevertheless, despite their success, transformers have two significant drawbacks that still limit their applicability in online reinforcement learning: (1) in order to remember all past information, the self-attention mechanism requires access to the whole history to be provided as context. (2) The inference cost in transformers is expensive. In this paper, we introduce recurrent alternatives to the transformer self-attention mechanism that offer context-independent inference cost, leverage long-range dependencies effectively, and performs well in online reinforcement learning task. We quantify the impact of the different components of our architecture in a diagnostic environment and assess performance gains in 2D and 3D pixel-based partially-observable environments (e.g. T-Maze, Mystery Path, Craftax, and Memory Maze). Compared with a state-of-the-art architecture, GTrXL, inference in our approach is at least 40% cheaper while reducing memory use more than 50%. Our approach either performs similarly or better than GTrXL, improving more than 37% upon GTrXL performance in harder tasks.

## Installation
Follow pip install -U "jax[cuda12]" for installing Jax and Jaxlib. Then run the following command to install the dependencies:
```
# Python version 3 is required
$ pip install -r requirements.txt
```
Install Weights and Biases for logging from https://docs.wandb.ai/quickstart.

## Usage
### T-Maze:
```
# AGaLiTe in T-Maze corridor length 160
$ python trainer.py +tmaze=agalite task.corridor_len=160

# GaLiTe
$ python trainer.py +tmaze=galite

# GTrXL256
$ python trainer.py +tmaze=gtrxl128

# GTrXL128
$ python trainer.py +tmaze=gtrxl256 

# LSTM
$ python trainer.py +tmaze=lstm

# GRU
$ python trainer.py +tmaze=gru
```

### Mystery Path:
```
# AGaLiTe (\eta=4) in MPGrid
$ python trainer.py +mysterypath=agalite4 task.env_name=MysteryPath-Grid-Easy-v0

# AGaLiTe (\eta=4) in MP
$ python trainer.py +mysterypath=agalite4 task.env_name=MysteryPath-Easy-v0

# GTrXL128 in MPGrid
$ python trainer.py +mysterypath=gtrxl128 task.env_name=MysteryPath-Grid-Easy-v0

# GTrXL64 in MP
$ python trainer.py +mysterypath=gtrxl64 task.env_name=MysteryPath-Easy-v0

# GTrXL32 in MP
$ python trainer.py +mysterypath=gtrxl32 task.env_name=MysteryPath-Easy-v0
```

## Available configurations:
The training script uses Hydra configuration management, the list of available configurations could be invoked using: 

```
$ python3 trainer.py +<TASK_NAME>=<BASE_CONFIG_NAME> --help
```

## Implementations
1. AGaLiTe implementation in Jax+Flax: `./src/models/galite/agalite.py`
2. GaLiTe implementation in Jax+Flax: `./src/models/galite/galite.py`
3. GTrXL implementation in Jax+Flax: `./src/models/gtrxl.py`

## Authors: 
1. Subhojeet Pramanik
2. Esraa Elilemy
3. Marlos C. Machado
4. Adam White

## Citation: 
```
@article{
pramanik2024agalite,
title={{AG}aLiTe: Approximate Gated Linear Transformers for Online Reinforcement Learning},
author={Subhojeet Pramanik and Esraa Elelimy and Marlos C. Machado and Adam White},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=lh6vOAHuvo},
note={}
}
```

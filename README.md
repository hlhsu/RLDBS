[![License](https://img.shields.io/github/license/analysiscenter/pydens.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://python.org)

# RL-DBS
# Reinforcement learning for suppression of collective neuronal activity in Deep Brain Stimulation (DBS)

This is a convenient gym environment for developing and comparing interaction of RL agents with several types of synthetic neuronal models of pathological brain activity. The ODEs that simulate synchronized neuronal signaling are wrapped into the the framework as individual environments, allowing simple switching between different physical models, and enabling convenient approbation of various RL approaches and multiple agents. The policy gradient algorithm PPO is shown to provide a robust data-driven control of neronal synchoryzation, agnostic of the neuronal model. 




### Installation as a project repository:

```
sudo apt-get install python3.8
pip3 install numpy
pip3 install pandas
pip3 install gym
pip3 install matplotlib
pip3 install stable-baselines3

git clone https://github.com/hlhsu/RLDBS.git
```

In this case, you need to manually install the dependencies.


### Examples:

```
cd RLDBS
python3 baseline_model_torch.py
```


### Important notes:

Environment uses generic Gym notation. A class that describes all relevant information is:
```
gym_oscillator/envs/osc_env.py
```


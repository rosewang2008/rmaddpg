# Recurrent MADDPG for Partially Observable and Limited Communication Settings


Code for [this paper](https://sites.google.com/view/rmaddpg/)

Presented at ICML 2019 RL for Real Life Workshop

## Known dependencies 
Before starting, please make sure you've satisifed the following reqs: 


```
baselines
```

Please also: 
```
cd maddpg && pip3 install -e . && cd ..
cd multiagent-particle-envs && pip3 install -e .
```


## Try it out
In order to run the algorithm you'll need to be in the `rmaddpg/maddpg/experiments` directory. In this directory, there is a `train.py` file which will allow you, among other things, try out the recurrent critic, recurrent actor, or both the recurrent actor-critic (R-MADDPG) model.

```
cd maddpg/experiments
python3 train.py --scenario simple_ --seed 1 --critic-lstm --actor-lstm
```

#!/usr/bin/env python3
import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
import maddpg.common.tf_util as U
from train import get_trainers, make_env
import argparse
from itertools import product

# TODO:
# Mouse clicking for changing location for a given agent (fixing all other positions)
# Keyboard event handling for changing location for agents
# Find the reward that cause the agents to go in different positions


def save_as_pickle(data, names, filename, force_save=False, just_get_info=True):
    """
    Both should be lists
    """
    if force_save: # data already contains info
        info = data
    else:
        info = {}
        for i, name in enumerate(names):
            info[name] = data[i]

    if just_get_info:
        return info
    else:
        with open(filename, 'wb') as fp:
            pickle.dump(info, fp)

    return info

def plot(data, xlabel='episodes', ylabel='time'):
    num_axes = len(data)
    fig, axs = plt.subplots(num_axes,1, constrained_layout=True)

    for i, p in enumerate(data.items()):
        axs[i].plot(p[1])
        axs[i].set_title(p[0])
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_ylim(0, 51)

    plt.show()

def get_position(env, name):
    if name == 'a1':
        return env.world.agents[0].state.p_pos
    elif name == 'a2':
        return env.world.agents[1].state.p_pos
    elif name =='goal':
        return env.world.landmarks[0].state.p_pos
    else:
        raise ValueError("[utils] Invalid name")
def save_and_close_plot(filename, descriptor):
    """ Saves currently open matplotlib plot under `filename`, and closes the plot.

    Parameters
    ----------
    filename : String
        Location to save the plot under.
        e.g. 'analysis/pictures/swapped_video_amax_step{}.{}'.format(i,'png')

    descriptor : String
        Short description of what the plot displays.

    Returns
    -------
    Nothing
        Just prints out that the method has saved the picture.

    """
    plt.savefig(filename)
    plt.close()
    print("Saved {} video analysis under {}".format(descriptor, filename))

def read(filename):
    infile = open(filename, 'rb')
    info = pickle.load(infile)
    infile.close()
    return info

def reset_tracking_info():
    return {'a1':[], 'a2':[], 'goal':[], 'random_positions':[]}

def create_random_positions(num=20):
    positions = []
    for i in range(20):
        ep = []
        for j in range(3):
            pos = np.random.uniform(-1.0,+1.0, 2)
            ep.append(pos)
        positions.append(ep)
    return positions

###
def create_save_random_positions(num=20):
    positions = create_random_positions(num=num)
    names = ['positions']
    save_as_pickle([positions],names,'random_positions')
    print("saved positions")
###


def create_obs_ph_n(n_agents, obs_shape_n):
    obs_ph_n = []
    for i in range(n_agents):
        obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
    return obs_ph_n

def create_dirs(arglist):
    if not os.path.exists(arglist.log_dir):
        os.makedirs(arglist.log_dir)
    if arglist.save_model and not os.path.exists(arglist.model_dir):
        os.makedirs(arglist.model_dir)

def log_info(writer, logs, episode, agent, value, arglist, str_tag, skip_log=False, sess=None):
    if not skip_log:
        logs[arglist.exp_name].info("Episode {}, agent {}: reward {:.5f}".format(episode, agent, value))
    if writer is None: return
    summary = tf.Summary(value=[tf.Summary.Value(tag=str_tag+"/agent_"+str(agent),
                                                    simple_value=value)])
    writer.add_summary(summary,episode)
    if sess:
        writer.add_graph(sess.graph)


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)

def set_log(arglist):
    log = {}
    set_logger(
        logger_name=arglist.exp_name,
        log_file=r'{0}/{1}.log'.format(arglist.log_dir, arglist.exp_name))
    log[arglist.exp_name] = logging.getLogger(arglist.exp_name)

    # Log arguments
    for (name, value) in vars(arglist).items():
        log[arglist.exp_name].info("{}: {}".format(name, value))
    return log

def get_lstm_states(_type, trainers):
    if _type == 'p':
        return [(agent.p_c, agent.p_h) for agent in trainers]
    elif _type == 'q':
        return [(agent.q_c, agent.q_h) for agent in trainers]
    else:
        raise ValueError("unknown type")


def create_action_ph(act_space_n, act_pdtype_n):
    return [act_pdtype_n[i].sample_placeholder([None]+[1], name="action"+str(i)) for i in range(len(act_space_n))]


def create_init_state(num_batches, len_sequence):
    c_init = np.zeros((1,len_sequence), np.float32)
    h_init = np.zeros((1,len_sequence), np.float32)
    return c_init, h_init


def get_lstm_state_ph(name='', n_batches=None, num_units=64):
    c = tf.placeholder(tf.float32, [n_batches, 1,  num_units], name=name+'c_ph')
    h = tf.placeholder(tf.float32, [n_batches, 1, num_units], name=name+'h_ph')
    return c, h


def update_trainers(trainers, writer, logs, episode_step, train_step, arglist, sess):
    loss = None
    for agent in trainers:
        agent.preupdate()
    for i, agent in enumerate(trainers):
        loss = agent.update(trainers, train_step)
        # print(loss)
        if (loss and not arglist.display): # [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
            log_info(writer, logs, episode_step, i, loss[0], arglist, "q_loss", skip_log=True, sess=sess)
            log_info(writer, logs, episode_step, i, loss[1], arglist, "p_loss", skip_log=True, sess=sess)
            log_info(writer, logs, episode_step, i, loss[2], arglist, "avg_q_target", skip_log=True, sess=sess)
            log_info(writer, logs, episode_step, i, loss[3], arglist, "train_reward", skip_log=True, sess=sess)
            log_info(writer, logs, episode_step, i, loss[4], arglist, "avg_next_q_target", skip_log=True, sess=sess)
            log_info(writer, logs, episode_step, i, loss[5], arglist, "std_q_target", skip_log=True, sess=sess)

def load_scenario(scenario_name):
    import multiagent.scenarios as scenarios
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    return scenario

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment_discrete import MultiAgentEnv
    import multiagent.scenarios as scenarios
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if "done" in dir(scenario):
        env = MultiAgentEnv(world, scenario.reset_world,
        scenario.reward, scenario.observation, done_callback= scenario.done, arglist=arglist)
    elif arglist.dual_reward:
        print("[DUAL REWARD] Using reward2 function!")
        env = MultiAgentEnv(world, scenario.reset_world,
        scenario.reward2, scenario.observation, arglist=arglist)
    elif benchmark:
        env = MultiAgentEnv(world, scenario.reset_world,
        scenario.reward, scenario.observation,
        scenario.benchmark_data, arglist=arglist)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
        scenario.reward, scenario.observation, arglist=arglist)
    return env


def store_reward(rew_n, writer, logs, episode_step, arglist, episode_rewards, agent_rewards):
    for i, rew in enumerate(rew_n):
        if writer is not None:
            log_info(writer, logs, episode_step, i, rew, arglist, "eval_reward", skip_log=True)
        episode_rewards[-1] += rew
        agent_rewards[i][-1] += rew

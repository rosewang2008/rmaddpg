import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import random
import analyze
import communication_tracker
import os
import rewards
import maddpg.common.tf_util as U
from maddpg.trainer.rmaddpg import MADDPGAgentTrainer , RMADDPGAgentTrainer, _RMADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import utils
import os

os.environ["CUDA_VISIBLE_DEVICES"]="3"



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--seed", type=int, default=1, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--update-freq", type=int, default=100, help="number of timesteps trainer should be updated ")
    parser.add_argument("--no-comm", action="store_true", default=False) # for analysis purposes
    parser.add_argument("--critic-lstm", action="store_true", default=False)
    parser.add_argument("--actor-lstm", action="store_true", default=False)
    parser.add_argument("--centralized-actor", action="store_true", default=False)
    parser.add_argument("--with-comm-budget", action="store_true", default=False)
    parser.add_argument("--analysis", type=str, default="", help="type of analysis") # time, pos, argmax
    parser.add_argument("--commit-num", type=str, default="", help="name of the experiment")
    parser.add_argument("--sync-sampling", action="store_true", default=False)
    parser.add_argument("--tracking", action="store_true", default=False)
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--test-actor-q", action="store_true", default=False)
    # Evaluation
    parser.add_argument("--graph", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--metrics-filename", type=str, default="", help="name of metrics filename")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64):
    # This model takes as input an observation and returns values of all actions
    print("Reusing MLP_MODEL: {}".format(reuse))
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def lstm_fc_model(input_ph, num_outputs, scope, reuse=False, num_units=64):
    print("Reusing LSTM_FC_MODEL: {}".format(reuse))
    with tf.variable_scope(scope, reuse=reuse):
        input_, c_, h_ = input_ph[:,:,:-2*num_units], input_ph[:,:,-2*num_units:-1*num_units], input_ph[:,:,-1*num_units:]
        out = input_
        out = layers.fully_connected(out, num_outputs=int(input_.shape[-1]), activation_fn=tf.nn.relu)
        c_, h_ = tf.squeeze(c_, [1]), tf.squeeze(h_, [1])
        cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
        state = tf.contrib.rnn.LSTMStateTuple(c_,h_)
        out, state = tf.nn.dynamic_rnn(cell, out, initial_state=state)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        c_, h_ = tf.expand_dims(state.c, axis=1), tf.expand_dims(state.h, axis=1) # ensure same shape as input state
        state = tf.contrib.rnn.LSTMStateTuple(c_,h_)
        return out, state

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, rewards.sim_higher_arrival_reward,  scenario.observation) # , done_callback=scenario.done)
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,  scenario.observation) # , done_callback=scenario.done)
    return env


def get_lstm_states(_type, trainers):
    if _type == 'p':
        return [agent.p_c for agent in trainers], [agent.p_h for agent in trainers]
    if _type == 'q':
        return [agent.q_c for agent in trainers], [agent.q_h for agent in trainers]
    else:
        raise ValueError("unknown type")

def update_critic_lstm(trainers, obs_n, action_n, p_states):
    obs_n = [o[None] for o in obs_n]
    action_n = [a[None] for a in action_n]
    q_c_n = [trainer.q_c for trainer in trainers]
    q_h_n = [trainer.q_h for trainer in trainers]
    p_c_n, p_h_n = p_states if p_states else [None, None]

    for trainer in trainers:
        q_val, (trainer.q_c, trainer.q_h) = trainer.q_debug['q_values'](*(obs_n + action_n + q_c_n + q_h_n))

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    trainer = _RMADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, mlp_model, lstm_fc_model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i,  mlp_model, lstm_fc_model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def create_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def train(arglist):
    # To make sure that training and testing are based on diff seeds
    if arglist.restore:
        create_seed(np.random.randint(2))
    else:
        create_seed(arglist.seed)

    with U.single_threaded_session() as sess:
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        if arglist.analysis:
            print("Starting analysis on {}...".format(arglist.analysis))
            if arglist.analysis != 'video':
                analyze.run_analysis(arglist, env, trainers)
            return # should be a single run

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        new_episode = True # start of a new episode (used for replay buffer)
        start_saving_comm = False

        if arglist.graph:
            print("Setting up graph writer!")
            writer = tf.summary.FileWriter("learning_curves/graph",sess.graph)

        print('Starting iterations...')
        while True:
            if arglist.actor_lstm:
                # get critic input states
                p_in_c_n, p_in_h_n = get_lstm_states('p', trainers) # num_trainers x 1 x 1 x 64
            if arglist.critic_lstm:
                q_in_c_n, q_in_h_n = get_lstm_states('q', trainers) # num_trainers x 1 x 1 x 64

            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            if arglist.critic_lstm:
                # get critic output states
                p_states = [p_in_c_n, p_in_h_n] if arglist.actor_lstm else []
                update_critic_lstm(trainers, obs_n, action_n, p_states)
                q_out_c_n, q_out_h_n = get_lstm_states('q', trainers) # num_trainers x 1 x 1 x 64
            if arglist.actor_lstm:
                p_out_c_n, p_out_h_n = get_lstm_states('p', trainers) # num_trainers x 1 x 1 x 64

            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                num_episodes = len(episode_rewards)
                # do this every iteration
                if arglist.critic_lstm and arglist.actor_lstm:
                    agent.experience(obs_n[i], action_n[i], rew_n[i],
                                    new_obs_n[i], done_n[i], # terminal,
                                    p_in_c_n[i][0], p_in_h_n[i][0],
                                    p_out_c_n[i][0], p_out_h_n[i][0],
                                    q_in_c_n[i][0], q_in_h_n[i][0],
                                    q_out_c_n[i][0], q_out_h_n[i][0], new_episode)
                elif arglist.critic_lstm:
                    agent.experience(obs_n[i], action_n[i], rew_n[i],
                                    new_obs_n[i], done_n[i], # terminal,
                                    q_in_c_n[i][0], q_in_h_n[i][0],
                                    q_out_c_n[i][0], q_out_h_n[i][0],new_episode)
                elif arglist.actor_lstm:
                    agent.experience(obs_n[i], action_n[i], rew_n[i],
                                    new_obs_n[i], done_n[i], # terminal,
                                    p_in_c_n[i][0], p_in_h_n[i][0],
                                    p_out_c_n[i][0], p_out_h_n[i][0],
                                    new_episode)
                else:
                    agent.experience(obs_n[i], action_n[i], rew_n[i],
                                    new_obs_n[i], done_n[i], # terminal,
                                    new_episode)

                obs_n = new_obs_n

            # Adding rewards
            if arglist.tracking:
                for i, a in enumerate(trainers):
                    if arglist.num_episodes - len(episode_rewards) <= 1000:
                        a.tracker.record_information("goal", np.array(env.world.landmarks[0].state.p_pos))
                        a.tracker.record_information("position",np.array(env.world.agents[i].state.p_pos))
                    a.tracker.record_information("ag_reward", rew_n[i])
                    a.tracker.record_information("team_dist_reward", info_n["team_dist"][i])
                    a.tracker.record_information("team_diff_reward", info_n["team_diff"][i])

            # Closing graph writer
            if arglist.graph:
                writer.close()
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                new_episode = True
                num_episodes = len(episode_rewards)
                obs_n = env.reset()
                # reset trainers
                if arglist.actor_lstm or arglist.critic_lstm:
                    for agent in trainers:
                        agent.reset_lstm()
                if arglist.tracking:
                    for agent in trainers:
                        agent.tracker.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
            else:
                new_episode=False

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None

            # get same episode sampling
            if arglist.sync_sampling:
                inds = [random.randint(0, len(trainers[0].replay_buffer._storage)-1) for i in range(arglist.batch_size)]
            else:
                inds = None

            for agent in trainers:
                # if arglist.lstm:
                #     agent.preupdate(inds=inds)
                # else:
                agent.preupdate(inds)
            for agent in trainers:
                loss = agent.update(trainers, train_step)
                if loss is None: continue

            # for displaying learned policies
            if arglist.display:
                env.render()
                # continue

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                # U.save_state(arglist.save_dir, saver=saver)
                if arglist.tracking:
                    for agent in trainers:
                        agent.tracker.save()

                rew_file_name = "rewards/" + arglist.commit_num + "_rewards.pkl"
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = "rewards/" + arglist.commit_num + "_agrewards.pkl"
                # agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

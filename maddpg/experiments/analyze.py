import pickle
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import utils
from itertools import product
import matplotlib.animation as animation
import communication_tracker

def stop():
    import sys; sys.exit(0)

def create_init_state(num_batches, len_sequence):
    """ Create an initial LSTM state of size (num_batches, 1, len_sequence)
    Parameters
    ----------
    num_batches : int
        Number of batches.
    len_sequence : int
        Length of the LSTM sequence.
    Returns
    -------
    2 values, both numpy array of size (num_batches, 1, len_sequence).
    First element is the cell state, the second is the hidden state.
    """
    c_init = np.zeros((num_batches, 1,len_sequence), np.float32)
    h_init = np.zeros((num_batches, 1,len_sequence), np.float32)
    return c_init, h_init

def track(env, terminal, done, policy):
    # episode done
    if terminal or done:
        # save all data
        utils.save_as_pickle(env.world.episode_info, None, 'rp_ep{}_{}'.format(env.world.position_index,policy), force_save=True)
        # reset
        env.world.episode_info = utils.reset_tracking_info()
        env.world.episode_info['a1'].append(env.world.positions[env.world.position_index+1][0].tolist())
        env.world.episode_info['a2'].append(env.world.positions[env.world.position_index+1][1].tolist())
        env.world.episode_info['goal'] = env.world.positions[env.world.position_index+1][-1].tolist()
        env.world.episode_info['random_positions'] = env.world.positions[env.world.position_index+1]
        return

    a1_pos = utils.get_position(env, 'a1').tolist()
    a2_pos = utils.get_position(env, 'a2').tolist()
    goal_pos = utils.get_position(env, 'goal').tolist()

    env.world.episode_info['a1'].append(a1_pos)
    env.world.episode_info['a2'].append(a2_pos)
    env.world.episode_info['goal']= goal_pos

def parse_info(info):
    pos_queries = info['pos_queries']
    q_vals = info['q_vals']
    test_positions = info["test_positions"]
    o = info['obs']
    a  = info['actions']
    argmaxs = info['argmaxs']
    return pos_queries, q_vals, test_positions, o, a, argmaxs

def correct_data(data):
    """ Reformats data to be a grid. Also corrects for rotation.
    Parameters
    ----------
    data : List of length num_queries. e.g. 1600.
        Contains results of queried information. e.g. argmaxs, q_vals, etc.
    Returns
    -------
    Numpy array of size (dim, dim), e.g. (40,40)
    """
    data = np.array(data)
    dim = int(math.sqrt(data.shape[0])) # (1600, 1)
    data = np.reshape(data, (dim,dim))
    data = np.rot90(data, 1) # need to rotate because of the way I'm querying Q function
    return data

def get_images(data, axs):
    images = []
    for i in range(len(axs.flat)-1):
        images.append(axs.flat[i].imshow(data[i])) # index into correct qvals
        axs.flat[i].label_outer()
    return images

def get_min_max(images):
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    return vmin, vmax

def set_titles(axs, names, main_title=""):
    plt.title(main_title)
    print("set_titles: names {}".format(names))
    for i, ax in enumerate(axs.flat):
        ax.set_title(names[i])

def compare_policies(policy1='patient', policy2='all_obs', episodes=20):
    for i in range(1, episodes):
        info_policy1 = utils.read(filename='rp_ep{}_{}'.format(i, policy1))
        info_policy2 = utils.read(filename='rp_ep{}_{}'.format(i, policy2))

        goal_pos = info_policy1['goal']

        print("goals {} {}".format(info_policy1['goal'],info_policy2['goal']))
        a1_policy1 = info_policy1['a1']
        a2_policy1 = info_policy1['a2']
        a1_policy2 = info_policy2['a1']
        a2_policy2 = info_policy2['a2']

        for j in range(len(a1_policy1)):
            fig, axs = plt.subplots(figsize=(16,7.5), ncols=2)
            plot_map_on_ax(a1_policy1[j], a2_policy1[j], goal_pos, axs[0])
            plot_map_on_ax(a1_policy2[j], a2_policy2[j], goal_pos, axs[1])
            names= [policy1,policy2]
            set_titles(axs, names)
            plt.savefig('analysis/pictures/rp_{}_{}_ep{}_timestep{}'.format(policy1, policy2, i,j))
            plt.close()


def create_color_bar(plt,axs, images, norm):
    if norm:
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("left", "5%", pad="90%")
        plt.colorbar(images[0], cax=cax,ax=axs[0])
    else:
        for i, im in enumerate(images):
            div = make_axes_locatable(axs[i])
            cax = div.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)
            cbar.ax.set_ylabel('Counts')

def normalize_images(images, vmin, vmax):
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

class InfoBag:
    def __init__(self, env, arglist, trainers, analysis_type):
        """
        Creates a bag filled with analysis information for analyzing and graphing.
        By assumption, my_agent will be trainers[0] and other_agent will be trainers[1]

        Had functionality for graphing:
            - argmax agent for the fixed agent.
            - q_value analysis for fixed agent.

        Can vary analysis view by:
            - fixing my_agent and varying other_agent's position
            - fixing other_agent and varying my_agent's position

        Parameters
        ----------
        env : type
            Description of parameter `env`.
        arglist : type
            Description of parameter `arglist`.
        trainers : type
            Description of parameter `trainers`.
        test_pos : type
            Description of parameter `test_pos`.
        analysis_type : type
            Description of parameter `analysis_type`.

        Returns
        -------
        Nothing, just displays desired plots.

        """

        # Settings
        self.min_bound = -1
        self.max_bound = 1

        self.env = env
        self.arglist = arglist
        self.trainers = trainers # [*trainers]


        if arglist.analysis == 'metrics':
            self._plot_metrics(arglist.metrics_filename)
            return

        # Different positions for agent 0
        diff_pos = [np.array([-1,-1]), np.array([1,-1]), np.array([-1,1]), np.array([1,1]),
                np.array([-0.75,-0.75]), np.array([0.75,-0.75]), np.array([-0.75,0.75]), np.array([0.75,0.75]),
                0.5*np.array([-1,-1]), 0.5*np.array([1,-1]), 0.5*np.array([-1,1]), 0.5*np.array([1,1]),
                0.25*np.array([-1,-1]), 0.25*np.array([1,-1]), 0.25*np.array([-1,1]), 0.25*np.array([1,1]),
                0.1*np.array([-1,-1]), 0.1*np.array([1,-1]), 0.1*np.array([-1,1]), 0.1*np.array([1,1])]

        for i in range(len(diff_pos)):
            self.test_pos = [diff_pos[i],
                            np.array([0.7,0.7]),
                            np.array([0,0])] # test_pos # [*agent_pos's, goal]
            assert len(self.trainers) == len(self.test_pos[:-1]), "Number of trainers must match with the number of test positions!"
            self.analysis_type = analysis_type # argmax, pos

            if not arglist.commit_num:
                print("NEED A COMMIT NUMBER FOR IDENTIFYING ANALYSIS")
                stop()
            self.filename = "visualizations/{}/{}_{}".format(arglist.commit_num, analysis_type, self.test_pos)

            # Print out the settings
            print("================================")
            print("[Analysis type]: {}".format(self.analysis_type))
            print("[Positions]:")
            for i in range(len(self.test_pos[:-1])):
                print("agent_{}: {}".format(i, self.test_pos[i]))
            print("goal: {}".format(self.test_pos[-1]))
            print("================================")

            # Data
            self.pos_queries = self._get_pos_queries() # 1600 x 2
            self.obs, self.swapped_obs = self._get_obs_queries(swap=False) # num_agents x 1600 x 1 x obs_dim
            self.actions, self.swapped_actions = self._get_actions(pick_action=False) # num_agents x 1600 x 1 x act_dim
            self.q_vals, self.swapped_q_vals = self._get_q_values(test_actor_q=False) # num_agents x 1600 x 1 x 1
            self.argmaxs, self.swapped_argmaxs = self._get_argmaxs(comm_argmax=False, u_dim=5) # num_agents x 1600 x 1


            # # # # Checking the dims
            # assert np.array(self.obs).shape == (len(self.trainers), len(self.pos_queries), 1, 9), "Got this as obs dim {}, but needed ({},{},1,{})".format(np.array(self.obs).shape, len(self.trainers), len(self.pos_queries), 9)

            # assert np.array(self.actions).shape == (len(self.trainers), len(self.pos_queries), 1, 7), "Got this as act dim {}, but needed ({},{},{}, 7)".format(np.array(self.actions).shape, len(self.trainers), len(self.pos_queries), 1)

            assert np.array(self.argmaxs).shape == (len(self.trainers), len(self.pos_queries), 1), "Got this as argmaxs dim {}, but needed ({},{},{})".format(np.array(self.argmaxs).shape, len(self.trainers), len(self.pos_queries), 1)

            # Save data
            self._save_info()

            # View analysis
            if arglist.analysis == 'pos' or arglist.analysis == 'argmax':
                self._view_analysis()
            else:
                self._view_all()
            # Terminate
        return

    def _get_pos_queries(self):
        xs, ys = np.arange(self.min_bound,self.max_bound,0.05), np.arange(self.min_bound,self.max_bound,0.05)
        pos_queries = list(product(xs,ys))
        return pos_queries

    def _get_pos(self, trainer_num, varied_pos):
        """Short summary.

        Parameters
        ----------
        trainer_num : type
            Description of parameter `trainer_num`.

        Returns
        -------
        Two agent positions in this order:
            trainers[trainer_num] position, other trainer position

        """
        # vary this agent's state, fix the other agent
        if len(self.trainers) == 1:
            return varied_pos, None
        else:
            if trainer_num == 0:
                fix_pos = self.test_pos[1] # the other agent is fixed
            else: # num = 1
                fix_pos = self.test_pos[0]

            return varied_pos, fix_pos

    def _get_obs_queries(self, swap=False):
        goal_pos = self.test_pos[-1]
        obs = []
        for i in range(len(self.trainers)):
            agent_obs = []
            for pos in self.pos_queries: # evaluating reference agent
                # if there's only one agent, fake the second position
                # i.e. returning varied_pos, None
                my_pos, other_pos = self._get_pos(i, pos)
                query_result = self._get_obs(my_pos, other_pos, goal_pos) # the varied position, goal pos
                agent_obs.append(query_result)
            obs.append(agent_obs)

        return obs, obs[::-1] # given obs, and swapped obs

    def _get_obs(self, my_pos, other_pos, goal_pos):
        # single agent
        # fixing the other agent's pos and vary mine
        my_dist = np.linalg.norm(my_pos-goal_pos)
        entity_pos = []
        for entity in self.env.world.landmarks:
            entity_pos.append(goal_pos - my_pos)

        if len(self.trainers) == 1:
            dist = None
        else:
            dist = np.linalg.norm(other_pos - goal_pos)

        # handles scenarios:
        #   - multiagent with no communication
        #   - single agent with no communication
        if self.arglist.no_comm or len(self.trainers)==1:
            communication = []
        else:
            communication = [np.array([1,dist])] # comm msg

        comm_budget = [np.array([1.0])] # dummy comm budget
        assert not communication if len(self.trainers)==1 else communication, "communication is invalid with single agent!"

        res = np.concatenate([my_pos] + entity_pos + communication +  comm_budget)
        # res = np.concatenate([my_pos] + [other_pos] +  [goal_pos] + comm_budget +  communication)
        res = np.array([res])
        return res

    def _get_actions(self, pick_action=False):
        if self.arglist.lstm:
            c, h = create_init_state(len(self.pos_queries), 64)
        if pick_action:
            raise NotImplementedError("Need to fix get_forced_actions implementation before using this functionality")
        actions = []
        swap_actions = []
        for i in range(len(self.trainers)):
            # action with normal obs
            if self.arglist.lstm:
                a_i = self.trainers[i].p_debug['target_act'](*([self.obs[i]] +[c] + [h] ))
            else:
                a_i = self.trainers[i].p_debug['target_act'](self.obs[i])
            actions.append(a_i)
            # action with swapped obs
            if self.arglist.lstm:
                a_i_swapped = self.trainers[i].p_debug['target_act'](*([self.swapped_obs[i]] +[c] + [h] ))
            else:
                a_i_swapped = self.trainers[i].p_debug['target_act'](*([self.swapped_obs[i]]))
            swap_actions.append(a_i_swapped)

        assert len(self.trainers) == len(actions), "Number of actions does not match up with number of trainers!"
        return actions, swap_actions

    def _get_forced_actions(self, obs, trainers, action, comm_on, agent, dim_u=5, dim_c=2):
        """
        2 * 5 variations
        - comm on or off
        - 0,1,2,3,4 physical actions
        comm_on {0,1}
        action {0,1,2,3,4}
        """
        size = len(obs[0])# number of observations
        u_action = np.zeros(dim_u)
        c_action = np.zeros(dim_c)

        u_action[action] = 1.0
        c_action[comm_on] = 1.0

        action = np.concatenate((u_action,c_action))
        forced_actions = np.repeat(action[None], size, axis=0)

        print("[force action] agent {}".format(agent))
        if agent == 0:
            return [forced_actions, trainers[1].p_debug['target_act'](np.array(obs[1]))]
        else:
            return [trainers[0].p_debug['target_act'](np.array(obs[0])), forced_actions]

    def _get_q_values(self, test_actor_q=False):
        q_vals = []
        swapped_q_vals = []
        c, h = create_init_state(num_batches=len(self.pos_queries), len_sequence=64)
        c_n = [c for i in range(len(self.trainers))]
        h_n = [h for i in range(len(self.trainers))]

        for agent in self.trainers:
            if not self.arglist.lstm:
                q_i = agent.q_debug['q_values'](*(self.obs+ self.actions))
                q_i_swapped = agent.q_debug['q_values'](*(self.swapped_obs + self.swapped_actions))
            elif test_actor_q:
                print("[analyze] Testing Q vals from Actor Network")
                q_i = agent.p_debug['testing'](*(self.obs + self.actions))
            else:
                q_i, _= agent.q_debug['q_values'](*(self.obs+ self.actions + c_n + h_n))
                q_i_swapped, _ = agent.q_debug['q_values'](*(self.swapped_obs + self.swapped_actions + c_n + h_n))
            q_vals.append(q_i)
            swapped_q_vals.append(q_i_swapped)

        assert len(self.trainers) == len(q_vals), "Number of q_vals does not match up with number of trainers!"
        return q_vals, swapped_q_vals

    def _get_argmaxs(self, comm_argmax=False, u_dim=5):
        """ Return the physical action index associated with the higher probability.
        If comm_argmax is true, then returns the highest communication action index,
        not the physical action.
        Parameters
        ----------
        actions : Numpy array of size (num_queries, 1, action_dim), e.g. (1600, 1, 7)
        comm_argmax: boolean
        If true, also considers the higher communiation action index.
        Else, ignores communication.
        u_dim: int
        Dimension of physical action distribution.
        Returns
        -------
        Numpy array of size (num_queries, 1) e.g. (1600, 1)
        Contains the indices associated with the highest action probability.
        """
        argmaxs = []
        swapped_argmaxs = []
        for i in range(len(self.trainers)):
            if comm_argmax:
                argmax_i = np.argmax(self.actions[i][:,:,u_dim:], axis=2)
            else:
                argmax_i = np.argmax(self.actions[i][:,:,:u_dim], axis=2)
                argmax_i_swapped = np.argmax(self.swapped_actions[i][:,:,:u_dim], axis=2)

            argmaxs.append(argmax_i)
            swapped_argmaxs.append(argmax_i_swapped)
        assert len(self.trainers) == len(argmaxs), "Number of argmaxs does not match up with number of trainers!"

        return argmaxs, swapped_argmaxs


    def _save_info(self):
        self.data = [self.pos_queries, self.test_pos,
                self.obs, self.actions,
                self.q_vals, self.argmaxs,
                self.swapped_obs, self.swapped_actions,
                self.swapped_q_vals, self.swapped_argmaxs]
        self.names = ["pos_queries", "test_positions",
                "obs", "actions",
                "q_vals", "argmaxs",
                "swapped_obs", "swapped_actions",
                "swapped_q_vals", "swapped_argmaxs" ]

        assert len(self.data) == len(self.names), "Number of data to be saved must be matched with the number of names to save data under!"

        print("Saving under filename: {} this information: {}".format(self.filename, self.names))
        # Creating directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        # plt.show()

        self.info = utils.save_as_pickle(self.data, self.names, self.filename)

    def get_vbounds(self,*argv):
        vmin = None
        vmax = None
        for arg in argv:
            arg = np.array(arg)
            vmin = arg.min() if (vmin is None or vmin > arg.min() ) else vmin
            vmax = arg.max() if (vmax is None or vmax < arg.max() ) else vmax
        return vmin, vmax

    def _view_all(self):
        # Create plot
        # Note:             obs = [from agent 1, from agent 2]
        #           swapped_obs = [from agent 2, from agent 1]
        #               actions = [policy1(obs1), policy2(obs2)]
        #       swapped_actions = [policy1(obs2), policy2(obs1)]
        #               q_vals  = [q_vals1(o1, a1), q_vals2(o2, a2)]
        #       swapped_q_vals  = [q_vals1(o2, p1(o2)), ...]
        #                       ==> just want to plot actions[0], and swapped_actions[1]
        #
        # Desired plot:
        #       q_value agent 1, swapped q value with agent 2   |
        #       same with argmax   | map

        fig, axs = plt.subplots(figsize=(9,6), nrows=2, ncols=3, constrained_layout=True)

        # plot q values
        q_vals = [correct_data(am) for am in self.info["q_vals"]]
        swapped_q_vals  = [correct_data(am) for am in self.info["swapped_q_vals"]]
        vmin, vmax = self.get_vbounds(q_vals,swapped_q_vals)

        psm = axs[0,0].pcolormesh(q_vals[0], vmin=vmin, vmax=vmax)
        fig.colorbar(psm, ax=axs[0,0])
        axs[0,0].set_aspect("equal","box")
        psm = axs[0,1].pcolormesh(swapped_q_vals[1], vmin=vmin, vmax=vmax)
        fig.colorbar(psm, ax=axs[0,1])
        axs[0,1].set_aspect("equal","box")

        # plot argmaxs
        argmaxs = [correct_data(am) for am in self.info["argmaxs"]]
        swapped_argmaxs  = [correct_data(am) for am in self.info["swapped_argmaxs"]]

        x = np.arange(self.min_bound,self.max_bound,2.0/40.0)
        y = np.arange(self.min_bound,self.max_bound,2.0/40.0)
        y = np.flip(y, axis=0) # bc matplotlib is weird with plotting

        im = axs[1,0].imshow(argmaxs[0])
        # self.annotate(x,y,axs[1,0],argmaxs[0])
        im = axs[1,1].imshow(swapped_argmaxs[1])
        # self.annotate(x,y,axs[1,1],swapped_argmaxs[1])

        # plot map
        self.plot_map_on_ax(self.test_pos, axs[0, -1])

        # naming plots
        names = ["qvals1(o1, p1(o1))", "qvals2(o1, p2(o1))", "map",
                "argmax(p1(o1))", "argmax(p2(o1))", "Empty"]

        set_titles(axs,names)

        # Save and display
        print("Saving plot under {}.png".format(self.filename))
        plt.savefig(self.filename+ ".png")
        # plt.show()

    def _plot_metrics(self, metrics_filename):
        # Metrics to display:
        #   reward, agreward, reward_mean, q_losses, p_losses,
        #   target_q_mean, target_q_next_mean, target_q_std
        with open(metrics_filename, "rb") as fp:
            self.metrics = pickle.load(fp)

        self.metrics_filename = metrics_filename
        # ==> batch by reward (3), loss (2), target vals (3)
        # label by agent number, plot info type, episode num
        self.agent_colors = ["r", "b"]

        # # # # # rewards
        fig, axs = self._create_plots(3,1,skip=True)
        self._populate_with_data(axs,["reward_mean"])
        self._populate_with_data(axs.flat[1:], ["final_ep_rewards", "final_ep_ag_rewards"], team=True)
        plt.savefig("visualizations/{}/_reward_metrics.png".format(self.arglist.commit_num))

        # # # # # loss
        fig, axs = self._create_plots(2,1)
        self._populate_with_data(axs,["q_losses","p_losses"])
        plt.savefig("visualizations/{}/_loss_metrics.png".format(self.arglist.commit_num))

        # # # # # target vals
        fig, axs = self._create_plots(3,1)
        self._populate_with_data(axs,["target_q_mean","target_q_std","target_q_next_mean"])
        plt.savefig("visualizations/{}/_target_metrics.png".format(self.arglist.commit_num))

    def _populate_with_data(self, axs, keywords, team=False):
        if team:
            for i, keyword in enumerate(keywords):
                d = self.metrics[keyword]
                axs[i].plot(np.array(d), c="g")
                axs[i].set_ylabel(keyword)
            return

        for i, keyword in enumerate(keywords):
            d = self._get_data(keyword)
            self._fill_subplot(axs, i, d)
            axs[i].set_ylabel(keyword)


    def _fill_subplot(self, axs, subplot_num, data):
        for i, color in enumerate(self.agent_colors):
            axs[subplot_num].plot(data[i], c=color)

    def _create_plots(self, rows, cols,skip=False):
        # bounds on episodes
        min_episode = self.metrics["timestamps"][0][0] # [first ep][get episode num]
        max_episode = self.metrics["timestamps"][-1][0]

        fig, axs = plt.subplots(rows, cols)
        fig.suptitle(self.metrics_filename)
        if skip:
            return fig, axs
        for ax in axs:
            ax.set_xlim(min_episode, max_episode)
            ax.set_xlabel("episodes")
        return fig, axs

    def _get_data(self, keyword):
        keyword_data = self.metrics[keyword]
        res = [np.array(keyword_data[i::2]) for i in range(len(self.trainers))]

        # make sure the data size is the same for both agents
        assert res[0].size == res[1].size, "expected data size to be the same by got sizes: {} vs {}".format(res[0].size, res[1].size)

        return res

    def _view_analysis(self, norm=False):
        # info = utils.read(filename=self.filename)
        if self.analysis_type == 'pos':
            self.create_heatmap(self.info, norm=norm, description="Q Val analysis")
        elif self.analysis_type == 'argmax':
            self.create_annotated_grid(self.info, description="Argmax analysis")

    def create_annotated_grid(self, info, description="", plot=False, swapped_policy=None):
        pos_queries, q_vals, test_positions, o, a, argmaxs = parse_info(info)
        data = [correct_data(argmaxs[i]) for i in range(len(argmaxs))]

        names = ["argmax for AGENT_{}".format(i) for i in range(len(argmaxs))] + ["map"]
        num_plots = len(names) # note, `map` is in names, not data
        fig, axs = plt.subplots(figsize=(16,7.5), ncols=num_plots)
        fig.suptitle(description)
        # create plots
        images = get_images(data, axs)

        x = np.arange(self.min_bound,self.max_bound,2.0/40.0)
        y = np.arange(self.min_bound,self.max_bound,2.0/40.0)
        y = np.flip(y, axis=0) # bc matplotlib is weird with plotting

        self.create_annotations(x,y,axs,data)
        self.plot_map_on_ax(test_positions, axs.flat[-1])
        set_titles(axs, names)

        fig.tight_layout()

        plt.savefig(self.filename+ ".png")
        if plot:
            plt.show()

    def plot_map_on_ax(self, test_positions, ax):
        colors = ['r', 'b', 'g']
        for i in range(len(test_positions[:-1])): # all positions except goal location
            # get x pos, y pos, cololr
            ax.scatter(test_positions[i][0], test_positions[i][1], c=colors[i])
        # adding goal location
        ax.scatter(test_positions[-1][0], test_positions[-1][1], c=colors[-1])

        color_names = ['red', 'blue', 'green']
        handles = []
        for i in range(len(test_positions[:-1])):
            patch = mpatches.Patch(color=color_names[i], label='AGENT_{}'.format(i))
            handles.append(patch)
        # adding goal patch
        handles.append(mpatches.Patch(color='green', label='goal'))

        ax.legend(handles=handles,loc='center left',bbox_to_anchor=(1,0.5))
        ax.set_xlim(self.min_bound,self.max_bound)
        ax.set_ylim(self.min_bound,self.max_bound)
        ax.set_aspect('equal','box')

    def create_heatmap(self, info, norm=False, description="", video=False, plot=False):
        pos_queries, q_vals, test_positions, o, a, argmaxs = parse_info(info)
        data = [correct_data(q_vals[i]) for i in range(len(q_vals))]

        # get data and names
        names = ["Q Values for AGENT_{}".format(i) for i in range(len(q_vals))] + ["map"]
        num_plots = len(names) # note, `map` is in names, not data
        fig, axs = plt.subplots(figsize=(16,7.5), ncols=num_plots)
        fig.suptitle(description)
        # create plots
        images = get_images(data, axs)

        if norm:
            vmin, vmax = get_min_max(images)
            normalize_images(images, vmin, vmax)
            create_color_bar(plt, axs, images, norm=norm)

        else:
            create_color_bar(plt, axs, images, norm=norm)

        self.plot_map_on_ax(test_positions, axs.flat[-1])
        set_titles(axs, names)
        plt.tight_layout()
        plt.xlabel("x_pos")
        plt.ylabel("y_pos")

        plt.savefig(self.filename+ ".png")
        if plot:
            plt.show()

    def create_video_image(self, info):
        pos_queries, q_vals1, q_vals2, agent1_pos, agent2_pos, goal, o, a = parse_info(info)
        q_vals1, q_vals2 = correct_data(q_vals1), correct_data(q_vals2)
        image = plt.imshow(q_vals1, animated=True)
        return image

    def annotate(self, x,y,ax, data):
        for i in range(len(y)):
            for j in range(len(x)):
                text = ax.text(j, i, data[i, j],
                               ha="center", va="center", color="w")
        r = mpatches.Patch(color='red', label='1 = east')
        b = mpatches.Patch(color='blue', label='2 = west')
        g = mpatches.Patch(color='green', label='3 = north')
        d = mpatches.Patch(color='red', label='4 = south')
        ax.legend(handles=[r,b,g,d],loc='center left',bbox_to_anchor=(1,0.5))
        # ax.set_xlim(self.min_bound,self.max_bound)
        # ax.set_ylim(self.min_bound,self.max_bound)
        # ax.set_aspect("equal", "box")


    def create_annotations(self, x,y,axs, data):
        for a, ax in enumerate(axs[:-1]):
            if a == len(axs.flat): break
            for i in range(len(y)):
                for j in range(len(x)):
                    text = ax.text(j, i, data[a][i, j],
                                   ha="center", va="center", color="w")
            r = mpatches.Patch(color='red', label='1 = east')
            b = mpatches.Patch(color='blue', label='2 = west')
            g = mpatches.Patch(color='green', label='3 = north')
            d = mpatches.Patch(color='red', label='4 = south')
            ax.legend(handles=[r,b,g,d],loc='center left',bbox_to_anchor=(1,0.5))
            # ax.set_xlim(self.min_bound,self.max_bound)
            # ax.set_ylim(self.min_bound,self.max_bound)
            # ax.set_aspect("equal", "box")


def run_analysis(arglist, env, trainers,u_dim=5, c_dim=2):
    # Analysis on Q function
    # dist_queries = [0.0]
    # test_pos = [np.array([0,0]),np.array([0,0])] # test_pos[-1] = goal, rest are agent locations

    if arglist.analysis == 'time':
        data = utils.read('time_analysis')
        utils.plot(data)
    elif arglist.analysis in {'pos', 'argmax', 'all', 'metrics'}:
        infobag = InfoBag(env, arglist, trainers, analysis_type=arglist.analysis)
    else:
        raise ValueError("Invalid analysis argument!")

def view_analysis(arglist, env, trainers,u_dim=5, c_dim=2, norm=False, dist_queries=[0.0]):
    if arglist.view_analysis == 'video':
        if False:
            images = []
            fig = plt.figure()
            for i in range(arglist.max_episode_len):
                info = utils.read(filename='video_step{}'.format(i))
                im = create_video_image(info)
                images.append([im])
            ani = animation.ArtistAnimation(fig, images, interval=arglist.max_episode_len, blit=True, repeat_delay=1000)
            plt.show()
        elif True: # this is for single policy evaluation
            for i in range(arglist.max_episode_len):
                description = 'video sequence at step {}'.format(i)
                info = utils.read(filename='video_step_swapped{}'.format(i))

                create_heatmap(info, norm=norm, description=description, plot=False)
                filename = 'analysis/pictures/swapped_video_q_step_{}.{}'.format(i,'png')
                utils.save_and_close_plot(filename=filename, descriptor='swapped_q_vals')

                create_annotated_grid(info,description=description, plot=False)
                filename = 'analysis/pictures/swapped_video_amax_step{}.{}'.format(i,'png')
                utils.save_and_close_plot(filename=filename, descriptor='swapped_argmax')

        elif False: # this is for swapped policy evaluation
            for i in range(arglist.max_episode_len):
                description = 'video sequence at step {}'.format(i)
                info = utils.read(filename='video_step{}'.format(i))
                swapped_info = utils.read(filename='swapped_video_step{}'.format(i))

                create_annotated_grid(info, description=description, plot=False, swapped_policy=swapped_info)
                filename = 'analysis/pictures/video_q_step{}.{}'.format(i,'png')
                utils.save_and_close_plot(filename=filename, descriptor='swapped_q_vals')
        else:
            raise ValueError("Nothing to analyze; check code")
    else:
        raise ValueError("Invalid analysis argument!")
    return

import numpy as np
import tensorflow as tf
import maddpg.common.tf_util as U
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

class CommunicationTracker:
    def __init__(self,agents, ep_length, filename="communication_tracker.pkl"):
        self.max_episode_length = ep_length
        self.agents = agents
        self.filename = "tracker_information/"+filename
        self.communication_tracker = dict() # <ep_number>:[communication], <ep_number>_location:[[agent0 location], [agent1 location], [goal location]]
        self.episode_communication = [[] for a in agents] # an episode's communication
        self.episode_number = len(self.communication_tracker)
        self.episode_location = [[] for i in range(len(agents)+1)] # num agent + goal

    def record(self, agent_num, action, timestep, agent):
        comm = np.argmax(action[0][-2:])
        self.episode_communication[agent_num].append(comm)
        self.episode_location[agent_num].append(np.array(agent.state.p_pos))

    def new_episode(self, episode_num, prev_goal=None):
        if prev_goal is not None:
            self.episode_location[-1].append(prev_goal)

        self.communication_tracker["{}_location".format(episode_num)] = np.array(self.episode_location)
        self.communication_tracker["{}_communication".format(episode_num)] = np.array(self.episode_communication)

        # Save under filename
        with open(self.filename, "wb") as f:
            pickle.dump(self.communication_tracker,f)

        self.episode_number += 1

        # resetting communication tracker
        self.episode_location = [[] for i in range(len(self.agents)+1)]
        self.episode_communication = [[] for a in self.agents]

    def get_tracker_data(self):
        with open(self.filename, "rb") as f:
            data = pickle.load(f)
        return data

    def plot(self, last20=True):
        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

        # Get communication info
        # Note: data.keys()/2 bc data stores both communication and location in different keys for a single episode
        data = self.get_tracker_data()
        cap = 20 if len(data.keys())/2 > 20 else len(data.key())

        for z in range(cap,1,-1):

            if last20:
                ys = data[2500+int(len(data.keys())/2)-z][0] # agent 0
            else:
                ys = data[z][0]

            xs = np.arange(0,len(ys), 1)
            ax.bar(xs,ys,zs=z,zdir='y', alpha=0.7)

        ax.set_xlabel("Episode timestep")
        # ax.set_xlim3d(0,self.max_episode_length)
        ax.set_zlabel("Communication on/off")
        # ax.set_ylim3d(0,1)
        ax.set_ylabel("Episodes")
        # ax.set_zlim3d(0, len(data.keys()))
        plt.show()

    def plot_communication(self):
        # plotting communication on 2D map
        # data[episode][agent_num] = [communication across episode timesteps]
        data = self.get_tracker_data()
        num_episodes = 20
        start_ep = 2500 + int(len(data.keys())/2)
        agent = 0

        for y in range(self.max_episode_length): # episode length
            for x in range(20): # total number of episodes to display
                # agent 0's communication
                comm = data[2500+x][agent][y]
                if comm:
                    plt.scatter(x,y)
        plt.xlabel("Episode")
        plt.ylabel("Timestep")

        plt.show()

    def plot_locations(self, episode_num):
        # Episode num >= 2500
        assert episode_num  >= 2500, "Episode number has to be larger than 2500"
        # 2D scatter map
        # agent0 = red; agent1 = blue; goal = black
        # Transparency ~ 0 => oldest location
        # Transparency ~ 0.90 => latest location
        # Transparency = 1 => used communication
        # 0.9 / 100 episodes = 0.009
        alpha_factor = 0.009
        d = self.get_tracker_data()
        # Check that # comm info = # location info
        assert len(d[2501][0]) == len(d["2501_location"][0]), "Comm data length does not match with location data length" # just taking on example

        comm_info = d[episode_num] # communication for both agent 0 and 1
        location_info = d["{}_location".format(episode_num)]


        # Plot goal
        goal_location = location_info[-1][0]
        plt.scatter(goal_location[0], goal_location[1], c='black')
        plt.title("Map of communication at episode {}".format(episode_num))
        for i in range(self.max_episode_length):
            # For each timestep
            # plot agent location
            for agent in range(len(comm_info)):
                if agent == 0 :
                    color = 'red'
                else:
                    color = 'blue'

                if comm_info[agent][i]:
                    if agent == 0:
                        color = 'yellow'
                    else:
                        color = 'green'
                    plt.scatter(location_info[agent][i][0], location_info[agent][i][1], c=color, alpha=1.0)
                else:
                    plt.scatter(location_info[agent][i][0], location_info[agent][i][1], c=color, alpha=alpha_factor*i)

        plt.show()

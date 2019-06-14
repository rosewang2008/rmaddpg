import numpy as np
import pickle


# For tracking information for each agent:
# - overall reward [should average]
# - per agent reward: split between own distance and joint reward [should average]
# - other metrics from maddpg_.py
# - time/distance to goal [should average]
# - communication usage


class InfoTracker:
    def __init__(self, agent_name,  arglist):
        self.agent_name = agent_name
        self.arglist = arglist
        self.base_filename = "../experiments/tracker_information/{}_{}_".format(self.arglist.commit_num, self.agent_name)

        keys = ["ag_reward", # every episode
                "team_dist_reward", # every episode
                "team_diff_reward", # every episode
                "q_loss", # every 100 timesteps
                "p_loss", # every 100 timesteps
                "target_q_mean", # same
                "reward_mean", # same
                "target_q_next_mean", # same
                "target_q_std", # same
                "communication",
                "position",
                "goal"
                ] # every episode
        # episode information
        self.episode_information = {key:[] for key in keys}

        # final tracked information
        self.final_information = {key:[] for key in keys}

        # start tracking bool
        self.start_tracking = False

    def start(self):
        self.start_tracking = True

    def record_information(self, kw, value):
        if not self.start_tracking: return
        self.episode_information[kw].append(value)

    def save(self):
        for k in self.final_information.keys():
            f = self.base_filename + k + ".pkl"
            print("Saving at {}".format(f))
            with open(f, "wb") as f:
                pickle.dump(self.final_information[k], f)
        print("Done saving everything")

    def reset(self):
        for k in self.final_information.keys():
            if "reward" in k:
                data = self.average(k)
                self.final_information[k].append(data)
            else:
                data = np.array(self.episode_information[k])
                self.final_information[k].append(data)

        self.episode_information = {key:[] for key in self.episode_information.keys()}


    def average(self, kw):
        return np.mean(self.episode_information[kw])


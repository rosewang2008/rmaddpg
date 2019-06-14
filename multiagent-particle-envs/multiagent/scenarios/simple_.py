import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

def create_init_state(num_batches, len_sequence):
    c_init = np.zeros((1, 1,len_sequence), np.float32)
    h_init = np.zeros((1, 1,len_sequence), np.float32)
    return c_init, h_init

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # Special
        world.damping = 0.0
        world.dim_c = 1
        world.communication_budget = 1.0 # n_agents * episode_length
        world.collaborative = True
        world.discrete_action = True # forcing discrete action in environment_discrete
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # agent properties
        world.agents[0].color = np.array([0.75,0.25,0.25])
        world.agents[1].color = np.array([102.0/255,178.0/255,1.0])
        world.communication_budget = 1.0

        # landmark properties
        world.landmarks[0].color = np.array([0.25,0.25,0.25])
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)


    def reward(self, agent, world):
        cost = 0.0
        for a in world.agents:
            # Penalize on distance
            cost -= np.sum(np.square(a.state.p_pos - world.landmarks[0].state.p_pos))

        # add to tracker
        print("adding team dist")
        agent.tracker.record_information("team_dist_reward", np.array(cost))

        other_agent = world.agents[0] if world.agents[0].name != agent.name else world.agents[1]
        dist1= np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)
        dist2= np.linalg.norm(other_agent.state.p_pos - world.landmarks[0].state.p_pos)
        team_diff =  abs(dist1 - dist2)
        cost -= team_diff
        agent.tracker.record_information("team_diff_reward", np.array(team_diff))
        return cost


    def used_communication(self, agent, world):
        # only deduct if comm is on and I have budget
        world.communication_budget -= 0.01

    def done(self, agent, world):
        return np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos) < 0.05

    def observation(self, agent, world):
        # get reference to other agent
        other_agent = world.agents[0] if world.agents[0].name != agent.name else world.agents[1]
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # get communication
        communication = []
        for a in world.agents:
            if a.name == agent.name: continue # if it's the current agent, skip
            if a.state.c and world.communication_budget >= 0:
                dist = np.array([np.linalg.norm(a.state.p_pos - world.landmarks[0].state.p_pos)])
                # reduce budget
                self.used_communication(agent, world)
                # message
                communication.append(a.state.p_pos)
            else:
                dist = np.array([-1])
                communication.append(np.array([-1,-1]))

        # add obs on communication budget
        comm_budget = [np.array([world.communication_budget])]
        old_comm = [other_agent.state.c, dist]
        # Partial obs
        # obs = np.concatenate([agent.state.p_pos] + [world.landmarks[0].state.p_pos] + communication + comm_budget)
        # obs = np.concatenate([agent.state.p_pos] + entity_pos + communication + comm_budget)

        # Fully obs
        obs = np.concatenate([agent.state.p_pos] + [world.landmarks[0].state.p_pos] + [other_agent.state.p_pos] + [other_agent.state.p_pos] + comm_budget)
        # obs = np.concatenate([agent.state.p_pos] + [other_agent.state.p_pos] + [world.landmarks[0].state.p_pos]+ comm_budget + communication)
        # obs = np.concatenate([agent.state.p_pos] + [other_agent.state.p_pos] + [world.landmarks[0].state.p_pos]+ comm_budget + old_comm)
        return np.array([obs])

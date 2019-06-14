import numpy as np

# Defining different reward functions
# Want to observe different behavior
# Modeled for simple_.py reward
# Using: world.communication_budget, world.communication_count for evaluating



####################################################################################################
# Centralized reward
# Based on simultaneous arrival and getting to goal
def normal_reward(agent, world):
    cost = 0.0
    for a in world.agents:
        # Penalize on distance
        cost -= np.sum(np.square(a.state.p_pos - world.landmarks[0].state.p_pos))
    # Penalize on simultaneity
    other_agent = world.agents[0] if world.agents[0].name != agent.name else world.agents[1]
    dist1= np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)
    dist2= np.linalg.norm(other_agent.state.p_pos - world.landmarks[0].state.p_pos)
    cost -= abs(dist1 - dist2)
    return cost

def sim_higher_arrival_reward(agent, world):
    cost = 0.0
    team_dist = 0.0
    for a in world.agents:
        # Penalize on distance
         team_dist-= np.sum(np.square(a.state.p_pos - world.landmarks[0].state.p_pos))

    cost += team_dist

    # agent.tracker.record("team_dist_reward",cost)

    # Penalize on simultaneity
    other_agent = world.agents[0] if world.agents[0].name != agent.name else world.agents[1]
    dist1= np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)
    dist2= np.linalg.norm(other_agent.state.p_pos - world.landmarks[0].state.p_pos)

    team_diff = -2*abs(dist1-dist2)
    # agent.tracker.record("team_diff_reward",team_diff)
    cost += team_diff
    return cost, team_dist, team_diff


# based on simultaneous arrival, getting to goal, every comm use for this agent
def self_comm_use_reward(agent, world):
    cost = 0.0
    for a in world.agents:
        # Penalize on distance
        cost -= np.sum(np.square(a.state.p_pos - world.landmarks[0].state.p_pos))
    # Penalize on simultaneity
    other_agent = world.agents[0] if world.agents[0].name != agent.name else world.agents[1]
    dist1= np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)
    dist2= np.linalg.norm(other_agent.state.p_pos - world.landmarks[0].state.p_pos)
    cost -= abs(dist1 - dist2)

    # Penalize on using communication
    if agent.action.c:
    	cost *= 2
    return cost

# based on simultaneous arrival, getting to goal, overusing communication
def self_comm_overuse_reward(agent, world):
    cost = 0.0
    for a in world.agents:
        # Penalize on distance
        cost -= np.sum(np.square(a.state.p_pos - world.landmarks[0].state.p_pos))
    # Penalize on simultaneity
    other_agent = world.agents[0] if world.agents[0].name != agent.name else world.agents[1]
    dist1= np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)
    dist2= np.linalg.norm(other_agent.state.p_pos - world.landmarks[0].state.p_pos)
    cost -= abs(dist1 - dist2)

    # Penalize on overusing communication
    if agent.action.c and not(world.communication_budget):
    	cost *= 2
    return cost


####################################################################################################
# Decentralized communication
# based on overusing communication for all team agents
def team_comm_overuse(agent,world):
    cost = 0.0
    # Penalize on overusing communication
    for a in world.agents:
	    if a.action.c and not(world.communication_budget):
	    	cost -= 1
    return cost

# based on overusing communication for this agent
def individual_comm_overuse(agent,world):
    cost = 0.0
    # Penalize on overusing communication
    if agent.action.c and not(world.communication_budget):
    	cost -= 1
    return cost

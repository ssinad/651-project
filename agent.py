#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

# from utils import rand_in_range, rand_un
import numpy as np
from env import number_of_states, max_reward, min_reward
import pickle

N = number_of_states()
p = 0.1
gamma = 0.9
seed = 0
v = np.zeros(N)  # [0 for tmp in range(N)]
num = np.zeros(N)
states = None
rewards = None
dp = False
epsilon = 1


def GS():
    return (max_reward() - min_reward()) / (1 - gamma)

def choose_action():
    toss = np.random.uniform()
    if toss < p:
        action = 0
    else:
        action = 1
    return action

def agent_init():
    global states, rewards
    states = []
    rewards = []
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    #initialize the policy array in a smart way

def agent_start(state):
    global states, rewards
    """
    Hint: Initialize the variables that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    action = choose_action()
    states.append(np.copy(state))
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """
    global states, rewards
    # select an action, based on Q
    action = choose_action()
    states.append(np.copy(state))
    # noise = 0
    # if dp:
    #     noise = np.random.laplace(scale=GS() / epsilon)
    rewards.append(reward)
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global states, rewards, v, num

    rewards.append(reward)
    # do learning and update pi
    return_so_far = 0
    unique_states = set()
    returns = np.copy(v)
    for i in range(len(states) - 1, -1, -1):
        state = int(states[i][0])
        if state not in unique_states:
            unique_states.add(state)
            num[state] += 1
        return_so_far *= gamma
        return_so_far += rewards[i]
        returns[state] = return_so_far
    # v[states[i]] += 1 / N[states[i]] * (return_so_far - v[states[i]])
    v += (returns - v) / num
    
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global v
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        noise = 0
        # if dp:
        #     noise = np.random.laplace(size=v.shape, scale=GS() / epsilon)
        return v + noise
    else:
        return "I don't know what to return!!"


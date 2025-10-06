#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
import BlackJackAgent as BJ
import numpy as np
import os

agent_name = "agent1"


env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")    
obs, info = env.reset()

agent1 = BJ.BJAgent(env,
             discount       = 1.0,
             alpha          = 0.01,
             kappa          = 0.0,
             epsilon_lo     = 0.01,
             epsilon_hi     = 0.07,
             epsilon_period = 600000,
             model_name     = "Agent_1_epsilongreedy"
             )

#-------------------------------------------------------------
# Train
#
def trainThis():
    n_episodes = 10000000
    win_hist, TD2_hist = agent1.train( env, n_episodes, 
                                      verbose=True , 
                                      reports=True, 
                                      report_period=5000 )
    agent1.saveQtable()

    return agent1, win_hist, TD2_hist

if os.path.isfile( agent1.model_name + ".pickle" ):
    agent1.loadQtable(agent1.model_name)
    win_hist = agent1.training_hist
else:
    agent1.initQtable()
    agent1, win_hist, TD2_hist = trainThis()
    
#-------------------------------------------------------------
# Analysis
#

from matplotlib import pyplot as plt

plt.figure()
plt.plot( agent1.visitation )

plt.figure()
plt.plot( TD2_hist )

plt.figure()
plt.pcolor( np.argmax( agent1.Q_vals, axis=1 ).reshape(32-4, 11-1 , 2)[:,:,0] )

plt.figure()
plt.plot( agent1.Q_vals )

 


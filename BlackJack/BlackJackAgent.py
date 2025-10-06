#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
import numpy as np
from random import uniform as uniform
from random import randint as randi
import pickle

class BJAgent:
    
    def __init__(self, env,
                 discount       = 1.0,
                 alpha          = 0.01,
                 kappa          = 0.0,
                 epsilon_lo     = 0.01,
                 epsilon_hi     = 0.01,
                 epsilon_period = 10000,
                 model_name     = "model_0"
                 ):
        
        # Parameters and environment variables
        #
        obs, info = env.reset()
        nn =  env.observation_space #Discrete(n)
        self._S  = ( nn[0].n-4 , nn[1].n-1, nn[2].n ) #Extract extent of discrete space
        self.state_space  = 1
        for s in self._S:
            self.state_space  *= s
        self.action_space = env.action_space.n
        
        self.current_state = self.state(obs)
        
        # Hyperparameters
        #
        self.discount       = discount
        self.alpha          = alpha
        self.kappa          = kappa
        self.epsilon_lo     = epsilon_lo
        self.epsilon_hi     = epsilon_hi
        self.epsilon        = self.epsilon_hi
        self.epsilon_period = epsilon_period
        
        #self.Q_vals
        self.model_name = model_name
        self.hasQ = False
        
        """
        if self.model_name == "model_0":
            self.initQtable()
        elif os.path.isfile( self.model_name + ".pickle" ):
            self.loadQtable(self.model_name)
        else:
            self.initQtable()
        """

    #---------------------------------------------------------------------
    # Observation Interpretation
    # 
    def state( self, obs ):
        obs1  = ( obs[0]-4, obs[1]-1, obs[2] )
        state = np.unravel_index( np.ravel_multi_index( obs1, self._S), self.state_space)
        return state
    
    
    #---------------------------------------------------------------------
    # Q management methods
    # 
    def initQtable(self):
        self.Q_vals = np.zeros( (self.state_space, self.action_space) )
        self.visitation = np.zeros( self.state_space )
        self.winrate_hist = []
        self.TD2_hist = []
        self.hasQ = True
        
    def loadQtable(self, fname):
        with open( fname + ".pickle", 'rb' ) as f:
            data = pickle.load(f)
        self.Q_vals        = data["Q"]
        self.visitation    = data["N"]
        self.winrate_hist  = data["WH"]
        self.TD2_hist      = data["LH"]
        self.model_name    = fname
        self.hasQ = True
    
    def saveQtable(self):
        if self.hasQ:
            data = {
                "Q"  : self.Q_vals,
                "N"  : self.visitation,
                "WH" : self.winrate_hist,
                "LH" : self.TD2_hist
                }
            with open( self.model_name + ".pickle", 'wb' ) as f:
                pickle.dump( data, f )
            return True
        else:
            return False
        
    

    #---------------------------------------------------------------------
    # Action methods
    #
    def explorationFnc(self,state):
        return 0.0
        #return self.kappa / ( 1.0 + np.sqrt(self.visitation[state]) )
    
    def getGreedyAction(self, state):
        a = np.argmax( self.Q_vals[state,:] )    
        return a
    
    #def getExploratoryAction(self,state):
    #    a = np.argmax( self.Q_vals[state,:] + self.kappa*self.explorationFnc(state) )
    #    return a
    
    def getEpsilonAction(self, state):
        x = uniform(0.0,1.0)
        if x < self.epsilon:
            a = randi( 0, self.action_space-1 )
        else:
            a = self.getGreedyAction(state)
            #a = self.getExploratoryAction(state)
        return a
    
    #---------------------------------------------------------------------
    # Gameplay
    #
    def updateQ(self, state, action, reward, terminated, next_state ):
        # Temporal Difference Learning
        next_Q = (not terminated) * np.max( self.Q_vals[next_state,:] )
        TD     =  reward + self.discount*next_Q - self.Q_vals[state,action]  
        self.Q_vals[state,action] += self.alpha * ( TD + self.explorationFnc(state) )
        return TD[0]
        
    def train(self, env, n_episodes, verbose=False, reports=False, report_period=1000):
        wins = 0
        TD2 = 0
        #winrate_hist = []
        for episode in range(n_episodes):
            self.epsilon = max( self.epsilon_lo, self.epsilon_hi*(1-episode/self.epsilon_period) )
            obs, info = env.reset()
            self.current_state = self.state(obs)
            self.visitation[self.current_state] += 1
            done = False
    
            # play one episode
            while not done:
                action = self.getEpsilonAction(self.current_state)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = self.state(next_obs)
                
                # update the agent
                tTD = self.updateQ( self.current_state, action, reward, terminated, next_state)
                TD2 += tTD**2
        
                # update if the environment is done and the current obs
                done = terminated or truncated
                self.current_state = next_state
                self.visitation[self.current_state] += 1

            if reward>0:
                wins += 1
            if (episode % report_period == 0)and(episode != 0) :
                TD2 = TD2/report_period
                if reports:
                    self.winrate_hist.append(wins)
                    self.TD2_hist.append(TD2)
                if verbose: 
                    print("Win rate this period: %i/%i, this period avg TD=%f,  current epsilon=%f"%(wins, report_period, TD2, self.epsilon) )
                wins = 0
                TD2 = 0
                
        return self.winrate_hist, self.TD2_hist
        
    
        
if __name__ == "__main__":
    
    env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")    
    obs, info = env.reset()
 
    if 1:
        agent0 = BJAgent( env )   
        state  = agent0.state(obs)
        agent0.initQtable()
        print("Q vals shape: ", agent0.Q_vals.shape)
        print("Game state:   ", obs, state )
        print("Q vals reflect action space: " , agent0.Q_vals[state,:] )
        print("greedy action:", agent0.getGreedyAction(state) )

    print("")
    print("Can agent train?: ")
    
    n_episodes = 100
    agent0.epsilon_period = 50
    win_hist, TD2_hist = agent0.train( env, n_episodes, verbose=True , reports=True, report_period=10 )
    
    print("")
    print("Can save and load models?")
    print("Sample from Q:     ",  np.max( agent0.Q_vals ) )
    print("save")
    my_name = agent0.model_name
    agent0.saveQtable()
    print("delete by reinitializing, then load")
    agent0.initQtable()
    agent0.loadQtable( my_name )
    print("Q sample is same?: ",  np.max( agent0.Q_vals ))
    
    
    """
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot( win_hist )
    plt.figure()
    plt.plot( agent0.visitation )
    plt.figure()
    plt.plot( agent0.Q_vals )
    plt.figure()
    plt.pcolor( np.argmax( agent0.Q_vals, axis=1 ).reshape(2, 11-1, 32-4)[:,:,0] )
    """
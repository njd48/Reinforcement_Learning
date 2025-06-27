# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 21:02:58 2025

@author: nickd
"""

import numpy as np
import random
from matplotlib import pyplot as plt

import banditNormalR as b


class Agent :
    # kArms
    # Rcum
    # Q
    # N
    # epsilon
    # alpha
    
    def __init__(self, kArms):
        self.kArms   = kArms
        self.Rcum    = 0.0
        self.Q       = np.zeros(kArms)
        self.N       = np.zeros(kArms)
        self.epsilon = 0.0
        self.alpha   = 0.05
        
    def greedyA(self):        
        return np.argmax(self.Q)
    
    def randomA(self):
        return random.randint( 0, self.kArms-1 )
        
    def setInitialQ(self, num):
        self.Q = num * np.ones(self.kArms)
        
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
        
    def learn(self, bandit ):
        
        x = random.uniform(0,1)
        if ( x > self.epsilon ):
            a = self.greedyA()
        else:
            a = self.randomA()
            
        R = bandit.play(a)
        self.N[a] = self.N[a] + 1
        #self.Q[a] = self.Q[a] + 1/self.N[a] * ( R - self.Q[a] )
        self.Q[a] = self.Q[a] + self.alpha * ( R - self.Q[a] )
        self.Rcum = self.Rcum + R
        
    def dontLearn(self, bandit):
        a = self.greedyA()
        R = bandit.play(a)
        self.Rcum = self.Rcum + R
        
    def modelError(self,bandit):
        return np.linalg.norm( self.Q - bandit.expectedR(), ord=2 )
        
    
def armBanditRate( N ):
    x = np.linspace(0,N-1,N)
    y = np.exp( -N*( x-N/2. )**2 )
    z = np.random.rand(N)
    r = y + z
    
    while (np.max(r)>=0.5):
        r = r * 0.5
        
    return r
        
if (__name__ == "__main__"):
    
    N = 20
    
    bandit = b.Bandit(N)
    bandit.setRatesEqual(0.0)
    
    #plt.plot(bandit.expectedR())
    #plt.show()
    
    
    
    agent0  = Agent(N) # <-- stupid agent
    agent1  = Agent(N)
    agent2  = Agent(N)
    agent3  = Agent(N)
    
    initialQ = 5.0
    agent0.setInitialQ(initialQ)
    agent1.setInitialQ(initialQ)
    agent2.setInitialQ(initialQ)
    agent3.setInitialQ(initialQ)
    
    agent0.setEpsilon(0.0)
    agent1.setEpsilon(0.0)
    agent2.setEpsilon(0.05)
    agent3.setEpsilon(0.15)
    
    # Give the stupid agent some training time
    for i in range(0,5):
        agent0.learn(bandit)        
    agent0.Rcum = 0
    
    
    TrainingSteps = 500    
    
    R0 = np.zeros(TrainingSteps)
    R1 = np.zeros(TrainingSteps)
    R2 = np.zeros(TrainingSteps)
    R3 = np.zeros(TrainingSteps)
    
    for i in range(0,TrainingSteps):
        agent0.dontLearn(bandit)
        agent1.learn(bandit)
        agent2.learn(bandit)
        agent3.learn(bandit)
        bandit.evolve(0.05)
        R0[i] = agent0.Rcum
        R1[i] = agent1.Rcum
        R2[i] = agent2.Rcum
        R3[i] = agent3.Rcum
    
    print("R0 cumulative: ", R0[[range(0,4)]], R0[[range(-4,-1)]] )
    print("R1 cumulative: ", R1[[range(0,4)]], R1[[range(-4,-1)]] )
    print("R2 cumulative: ", R2[[range(0,4)]], R2[[range(-4,-1)]] )
    print("R3 cumulative: ", R3[[range(0,4)]], R3[[range(-4,-1)]] )
    print(" ")
    print("bandit opt action: ", np.argmax(bandit.expectedR())  )
    print("agent0 greedy a:   ", agent0.greedyA()  )
    print("agent1 greedy a:   ", agent1.greedyA()  )
    print("agent2 greedy a:   ", agent2.greedyA()  )
    print("agent3 greedy a:   ", agent3.greedyA()  )

    #plt.plot(bandit.expectedR())
    #plt.show()
    
    plt.plot( R0 )
    plt.plot( R1 )
    plt.plot( R2 )
    plt.plot( R3 )
    plt.legend({'agent 0','agent 1','agent 2','agent 3'})
    plt.show()
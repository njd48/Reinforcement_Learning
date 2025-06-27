# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 19:45:32 2025

@author: nickd
"""

import numpy as np
import random

class Bandit:
    #kArms  
    #cost   
    #payout 
    #rates  
    
    def setRatesRand(self):
        upperlim = 1./5.
        self.rates = np.random.rand(self.kArms)*upperlim
        
    def setRates(self, rates ):
        assert( rates.size==self.kArms )
        self.rates = rates
        
    def __init__(self, kArms, cost=1, payout=16):
        self.kArms  = kArms
        self.cost   = cost
        self.payout = payout
        self.rates  = np.zeros(kArms)
        self.setRatesRand()
    
        
    def expectedR(self):
        return self.payout * self.rates - self.cost
        
    def play( self, n ):
        x = random.uniform(0,1)
    
        if ( x > self.rates[n] ):
            return -self.cost
        else:
            return self.payout-self.cost
        
        
if ( __name__=='__main__' ):
    
    N = 10
    
    bandit = Bandit( N )
    
    print( bandit.rates )
    print( bandit.expectedR() )
    
    bandit.play(2)
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 19:45:32 2025

@author: nickd
"""

import numpy as np
from matplotlib import pyplot as plt

class Bandit:
    # kArms 
    # payE  array, expected payout
    # payS  array, stdev of payout
    
    def setRatesEqual(self, cnst = 1):
        self.payE = cnst * np.ones(self.kArms)
        self.payS = cnst * np.ones(self.kArms)
    
    def setRatesRand(self, breadth = 1):
        self.payE = (breadth)*(-1 + 2*np.random.rand(self.kArms))
        self.payS = (breadth)*( np.random.rand(self.kArms) )
        
    def setExpectedR(self, R ):
        assert( R.size==self.kArms )
        self.payE = R
        
    def setStDevR(self, S):
        assert( S.size==self.kArms )
        self.payS = S
        
    def __init__(self, kArms, paystyle="rand", payparam = 1, varparam = 0):
        self.kArms  = kArms
        
        match paystyle:
            case "rand":
                self.setRatesRand(breadth=payparam)
            case "even":
                self.setRatesRand(cnst=payparam)
            case "custom":
                self.setExpectedR(payparam)
                self.setStDevR(varparam)                
            case _:
                print("warning, bad case provided, defaulting to random payouts.")
                self.setRatesRand(breadth=payparam)
                
    #----------------------------------------------------------------------
        
    def expectedR(self):
        return self.payE
        
    def play( self, n ):
        x = np.random.normal( self.payE[n], self.payS[n] )    
        return round(x)
    
    def evolve(self,dr):     # Expected action values, and variance, are modified by random walk
        self.payE = np.add(  self.payE, 2*dr *(-0.5 + np.random.rand(self.kArms) ) )
        self.payS = np.multiply( self.payS, 1.0 + 2*dr*(-0.5 + np.random.rand(self.kArms) ) )
        
        
if ( __name__=='__main__' ):
    
    N = 10
    
    bandit = Bandit( N )
    
    print( bandit.expectedR() )
    
    bandit.play(2)
    
    plt.plot(bandit.expectedR())
    plt.show()
    
    
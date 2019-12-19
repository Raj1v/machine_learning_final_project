#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:08:25 2019

@author: Rajiv
"""

import random
import numpy as np


class RecommendRandom(object):
    def __init__(self):
        return

        
    def get_ratings(self, targetid, targetmovies):
        """Returns random ratings for each target movie"""
        
        predictions = []
        
        
        for movie in targetmovies:
            prediction = random.randrange(1, 6) # random number between 1 and 5
            predictions.append(prediction)
        
        return np.array([targetmovies, predictions]).T
    

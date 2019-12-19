#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:19:46 2019

@author: Rajiv
"""

import numpy as np

class RecommendMean(object):
    def __init__(self):
        return
        
    def get_ratings(self, targetid, targetmovies):
        """Returns mean ratings for each target movie"""    
        predictions = []   
        
        for movie in targetmovies:
            prediction = self.mean_of_movies[movie + 1]
            predictions.append(prediction)
        
        return np.array([targetmovies, predictions]).T
    
    def init_data(self, data):
        #Take the input data matrix and add two new columns at the start, the first with the user id, the second with 
        #this user's mean. Also if we want all our data to be normalized and mean=True, we do that here as well
        new_data = data.copy()
        transp_data = new_data.T
        mean_of_movies = []
        #calculate the mean of all_movies
        for i in range(len(transp_data)):
            movie_mean = new_data[i][new_data[i] != 0].mean()
            mean_of_movies.append(movie_mean)
        
        self.mean_of_movies = np.array(mean_of_movies)
        
    
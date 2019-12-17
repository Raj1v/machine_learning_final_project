#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:36:21 2019

@author: Rajiv
"""

# import math

from collections import defaultdict

import numpy as np

def get_init_matrix():
    lo_MovieId = []
    lo_movie = []
    
    #Here we import the movies.dat file, where we have a twodimensional matrix 
    #with all MovieID's and corresponding names
    
    with open("data/ml-1m/movies.dat", "r", encoding='windows-1252') as infile:
        for line in infile:
            movieId, movie, _ = line.split("::")
            lo_MovieId.append(movieId)
            lo_movie.append(movie)
    
    arr_MovieId = np.array(lo_MovieId)[:, np.newaxis]
    arr_movie = np.array(lo_movie)[:, np.newaxis]
    MovieId2movie = np.concatenate((arr_MovieId, arr_movie), axis=1)
    
    #First we import ratings.dat and put it in a dictionary
    user2movies_ratings = defaultdict(dict)
    los_movieId = []
    
    with open("data/ml-1m/ratings.dat", "r", encoding='utf-8') as infile:
        for line in infile:
            userId, movieId, rating = [int(el) for el in line.split("::")[:3]]
            user2movies_ratings[userId][movieId] = rating
            los_movieId.append(movieId)
    
    max_movieId = max(los_movieId)
    
    # Then we transform it into a data matrix where all users rate all movies, 
    # movies that are not rated get a value of 0. This matrix is called the init_matrix
    
    len_users = len(user2movies_ratings)
    len_movies = max_movieId
    
    # Jasper, dit is de data matrix: init_matrix
    init_matrix = np.zeros((len_users,len_movies))
    
    #Hier kijk ik of het voor mij makkelijk is als ik aan deze matrix nog twee colommen toevoeg met de user id en hun mean score
    #Ik weet nog niet zeker of ik dat ook zo ga gebruiken, maar zo ja dan doe ik dat in mijn code wel
    for user in user2movies_ratings:
        for movie in user2movies_ratings[user]:
            review = user2movies_ratings[user][movie]
            init_matrix[user-1, movie-1] = review
    
    all_id_means =[]
    for i in range(len_users):
        all_id_means.append([i+1, init_matrix[i][init_matrix[i] != 0].mean()])
        
    arridmeans = np.array(all_id_means)
    
    matrix_and_means = np.hstack((arridmeans,init_matrix))
    
    return init_matrix
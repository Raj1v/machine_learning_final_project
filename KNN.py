import math

from itertools import product, combinations
from operator import itemgetter
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas.compat import StringIO

from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error

class KNearestNeighbour(object):
    def __init__(self, k = 50, sim_treshold = 0, measure = "cosine", show = True, mean= True):
        self.k = k
        self.sim_treshold = sim_treshold
        self.smeasure = measure
        self.fmeasure = None
        self.data = None
        self.MovieId = None
        self.targetid = None
        self.compid = None
        self.neighbourhood = None
        self.prediction = None
        self.targetmovies = None
        self.mean = mean
        self.mean_of_movies = None
        self.show = show
        
        
    def init_data(self, data):
        #Take the input data matrix and add two new columns at the start, the first with the user id, the second with 
        #this user's mean. Also if we want all our data to be normalized and mean=True, we do that here as well
        new_data = data.copy()
        all_id_means =[]
        transp_data = new_data.T
        mean_of_movies = []
        #calculate the mean of all_movies
        for i in range(len(transp_data)):
            movie_mean = new_data[i][new_data[i] != 0].mean()
            mean_of_movies.append(movie_mean)
        
        self.mean_of_movies = np.array(mean_of_movies)
        
        #Add the userID and mean for that user to the matrix, if mean = True
        #also subtract that mean from all ratings
        for i in range(len(new_data)):
            id_mean = new_data[i][new_data[i] != 0].mean()
            all_id_means.append([i+1, id_mean])
            if self.mean:
                new_data[i][new_data[i] !=0] -= id_mean
        
        id_mean = np.array(all_id_means)
        self.data = np.hstack((id_mean,new_data))
        
    def get_neighbourhood(self):
        #Get all the movies that your target id has rated without the movies you want to target
        targetid_movies = np.setdiff1d((np.where(self.data[self.targetid] !=0)[0] -1),self.targetmovies) + 1
        
        #Get an array of all the ratings of your target
        targetid_ratings = self.data[self.targetid][targetid_movies]
        
        #Take only the targeted movies
        only_targets = self.data.T[targetid_movies].T
        
        #Be sure to take the data from only_targets[:,2:] as everyone has their id and mean in the first two columns
        no_empty_ids = only_targets[np.any(only_targets[:,2:]!=0,axis=1)]
        not_user = no_empty_ids[no_empty_ids[:,0]!= self.targetid+1]
        
        if self.show:
            print("Shape of targetid_movies",targetid_movies.shape,"\n")
            print("Shape of targetid_ratings",targetid_ratings.shape,"\n")
            print("Shape of only_targets",only_targets.shape,"\n")
            print("Shape of no_empty_ids",no_empty_ids.shape,"\n")
            print("Shape of not_user",not_user.shape,"\n")

        #Get the Matrix of users who have at least the sim_treshold amount of ratings        
        satisfy_n = []
        for i in range(len(not_user)):
            if len(not_user[i][not_user[i] != 0]) <= self.sim_treshold:
                continue
            satisfy_n.append(not_user[i])
        final_matrix = np.array(satisfy_n)
        
        if self.show:
            print("Shape of final_matrix is:",final_matrix.shape,"\n")
        
        #Now calculate the similarity for each id that has at least sim_treshold movies in common
        sim_and_id = []
        for compid in final_matrix:
            comp_movies = np.where(compid !=0)[0]
            target_ratings = targetid_ratings[comp_movies][2:]
            comp_ratings = compid[comp_movies][2:]
            sim = self.fmeasure(target_ratings, comp_ratings)
            sim_and_id.append([compid[0], sim])

        sim_and_id = np.array(sim_and_id)
        
        if self.show:
            print("Shape of sim_and_id:",sim_and_id.shape,"\n")
            print("Example of sim_and_id is:",sim_and_id[:5],"\n")
        
        #We define our neighbourhood to be the first k highest sims, with their ID's in the first column                       
        neighbourhood = sim_and_id[np.flip(np.argsort(sim_and_id.T[1, :]))][:self.k]
        self.neighbourhood = neighbourhood
        
    def get_recommendations(self):
        #Get the users who where in this neighbourhood, make the type int so we can use indexing
        top_k_users = self.neighbourhood.T[0].astype(int) - 1
        
        #Take the movie_ratings our target had, only used if show = True
        target_movie_ratings = self.data[self.targetid][self.targetmovies+1]
        
        #Takes the matrix of neighbour_ratings, for all our k users we get their scores regarding the target_movies
        neighbour_ratings = self.data[top_k_users][:,self.targetmovies+1]

        if self.show:
            print("Targetuser has these ratings: ",target_movie_ratings,"\n")
            print("top_k_users has shape: ",top_k_users.shape,"\n")
            print("neighbour_ratings[:5] looks like: ",neighbour_ratings[:5],"\n")
        
        #Calculate the predictions for each movie
        predictions = []
        for movie in neighbour_ratings.T:
            i = 0
            target_movie = self.targetmovies[i] -1
            i += 1
            
            users_rated = np.where(movie !=0)[0]
            
            if len(users_rated) == 0:
                #If there are no ratings for this movie we take this movie's mean
                movie_mean = self.mean_of_movies[target_movie]
                predictions.append(movie_mean)
                continue
            
            sims_users = self.neighbourhood.T[1][users_rated]
            weighted_scores = movie[users_rated] * sims_users
            predict = sum(weighted_scores) / sum(sims_users)
            if self.mean:
                predict = predict + self.data[self.targetid][1]
            predictions.append(predict)
          
        predictions = np.array(predictions)  
        self.predictions = predictions
      
        if self.show:
            print("predictions looks like ",predictions)
            
    def get_ratings(self, targetid, targetmovies):
        """Combines get_neighbourhood and get_recommendations to get the final predicted score per movie"""
        #As we are mostly working with indexes subtracted the 1 from a movie ID here, makes our life easier in the future
        self.targetid = targetid - 1
        self.targetmovies = targetmovies
        
        
        self.sim_measure()
        self.get_neighbourhood()
        self.get_recommendations()
        
        #Return an array of the movies and there respected score
        if self.show:
            print(np.array([self.targetmovies,self.predictions]).T)
        return np.array([self.targetmovies,self.predictions]).T
    
          
    def euclidean_similarity(self, p, q):
        dist = math.sqrt(sum((pi-qi)**2 for pi,qi in zip(p, q)))
        sim = 1 / (1+dist)
        return sim   
        
    def manhattan_similarity(self, p, q):
        dist = sum(np.abs(pi-qi) for pi,qi in zip(p, q))
        sim = 1 / (1+dist)
        return sim    

    def cosine_similarity(self, p, q):
        d = sum(pi * qi for pi,qi in zip(p, q))
        mag_p = math.sqrt(sum([pi**2 for pi in p]))
        mag_q = math.sqrt(sum([qi**2 for qi in q]))
        sim = d / ( mag_p * mag_q)
        return sim
    
    def pearson_correlation(self, p, q):
    # this code does not scale well to large datasets. In the following, we rely on 
    # scipy.spatial.distance.correlation() to compute long vectors
        if len(p) > 99:
            return 1 - distance.correlation(p,q)        
        
        p_mean = sum(p) / len(p)
        p_deviations = [(pi-p_mean) for pi in p]
        
        q_mean = sum(q) / len(q)
        q_deviations = [(qi-q_mean) for qi in q]
        
        cov = sum(pde * qd for pde,qd in zip(p_deviations, q_deviations))
            
        sds_product = math.sqrt(sum((pder)**2 for pder in p_deviations) * sum((qd)**2 for qd in q_deviations))
        
        if sds_product != 0:
            r = cov / sds_product
        else:
            r = 0
        return r
    
    def jaccard_sets(self, p, q): 
        intersection_cardinality = len(set(p).intersection(set(q)))
        union_cardinality = len(set(p).union(set(q)))
        sim = intersection_cardinality / union_cardinality
        return sim
    
    def jaccard_binary(self, p, q):
        # only works for binary vectors! Binarize your vectors first
        m_11, m_01, m_10 = 0, 0, 0
        for pi, qi in zip(p, q):
    
            if pi == 1:
                if qi == 1:
                    m_11 += 1
                else:
                    m_10 += 1
                    
            elif qi == 1:
                m_01 += 1
        
        sim = m_11 / (m_10 + m_01 + m_11) 
        return sim
    
    def sim_measure(self):
        all_measures = {"euclidean" : self.euclidean_similarity,
                        "manhattan" : self.manhattan_similarity,
                        "cosine": self.cosine_similarity, 
                        "correlation": self.pearson_correlation,
                        "jaccardset" : self.jaccard_sets,
                        "jaccardbin" : self.jaccard_binary}
        if self.smeasure in all_measures.keys():
            self.fmeasure = all_measures[self.smeasure]
            
# machine_learning_final_project
Recommendation system
Jasper, Lukas, Rajiv en Hanabi

For the project we will write an implementation of a recommendation system.

Which problem do you want to solve?
- Recommend 
Which paper(s) do you base this on? 

Which algorithms are you going to use?
- We will use the k-NN classifier and SVD for all unrated movies.
- Say every there are m users and n rated movies. Every user will have a data set which is an array of 
Where do you get the data from?
- We will use the MovieLens 1M dataset from the university of Minnesota. It contains 1 million reviews from 6040 users.

Topics to discuss
- cold data, how to start with a new user, which questions to ask to gather the most information.
- SVD
- K-NN classifier
  - which K are we going to use?
    number of samples (i.e. n_samples), 6040 users and dimensionality (i.e. n_features), 4000 movies.
    Brute force query time grows as O[DN]
    Ball tree query time grows as approximately O[DlogN]
    KD tree query time changes with  in a way that is difficult to precisely characterise. For small  (less than 20 or so) the cost is approximately O[DN], and the KD tree query can be very efficient. For larger , the cost increases to nearly, O[DN], and the overhead due to the tree structure can lead to queries which are slower than brute force.

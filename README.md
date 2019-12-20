# Machine Learning final project
Recommendation system

By Jasper, Lukas, Rajiv en Hanabi

For the project we will write an implementation of a recommendation system. Where the model will be trained to give a predicted review on a movie by a user. To do so the model will analyze the reviews of its nearest neighbours to predict a review. 

# project outline
Input data: We will use the MovieLens 1M dataset from the University of Minnesota. It contains 1 million reviews from 6040 users.
Create the data matrix 
User ids (6040)  against movies (3952)
Review from: 1 - 5, No review:  Empty
Fill in ratings in every cell (1 M), Fill in blank spaces (23 M)
### Mean center each user’s reviews
Subtract the mean of the user’s reviews from all its reviews
Since every user has a different way of rating, some more critical than others, we will normalize the data by using the mean-centered data. 
After the recommendation gives mean-centered predicted reviews we add the target user’s mean again to resemble the actual predicted review.
### Recommendation system
Target user to all other users matrix
Check which other users have reviewed some movies
From this point, we only look at those users
Mean-center normalize per user, incl the target user
Input (Data_matrix, target_user)

Apply K-NN to target user
Closer neighbors get higher weights
Weight = similarity divided by the total similarity of k nearest neighbors
Output: The predicted reviews of the target user

### Evaluation
Input: Data matrix, without one target review of the target movies
Target user
Predicted reviews
F-measure the results. What is the loss/cost?
Compare the predicted reviews from the recommendation system to the actual reviews in the data matrix
False positives more weight than false negatives
See which similarity measure performs best
- Cosine, Manhatten, Jaccard, Euclidean

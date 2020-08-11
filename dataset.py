# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data

print("\n\nWelcome to my recommendation system of MovieLens dataset!. \n"\
      "I didn't find any dataset available on Course recommendation system, so I decided to work on MovieLens dataset. \n"\
      "You can find this dataset on http://files.grouplens.org/datasets/movielens/ml-20m-README.html \n\n"\
      "Please Go through this program and give me your feedback if you like it or want any clarification on something."\
      ' I will highly recommend you to install all necessary libraries and use at least 8GB RAM to run this program.')
# Importing the dataset
print("\nReading Movies\n")
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
print("Reading Users\n")
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
print("Reading Ratings\n")
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

user_dataframe = pd.DataFrame()
user_dataframe['UserID'] = users[0]
user_dataframe['Gender'] = users[1]
user_dataframe['Age in Years'] = users[2]
occupation_number = list(users[3])
occupation_name = ["other", "academic/educator", "artist", "clerical/admin", "college/grad student", "customer service",
                   "doctor/health care", "executive/managerial", "farmer", "homemaker", "K-12 student", "lawyer",
                   "programmer", "retired", "sales/marketing", "scientist", "self-employed", "technician/engineer", 
                   "tradesman/craftsman", "unemployed", "writer"]
occupation = occupation_number[:]
for i in range(len(occupation_number)):
    for j in range(len(occupation_name)):
        if(occupation_number[i] == j):
            occupation[i] = occupation_name[j] 
user_dataframe['Occupation '] = occupation
user_dataframe['Pincode'] = users[4]
user_dataframe.to_csv('users.csv')

movie_dataframe = pd.DataFrame()
movie_dataframe['MovieID'] = movies[0]
movie_dataframe['Title'] = movies[1]
movie_dataframe['Genres'] = movies[2]
movie_dataframe.to_csv('movies.csv')

rating_dataframe = pd.DataFrame()
rating_dataframe['UserID'] = ratings[0]
rating_dataframe['MovieID'] = ratings[1]
rating_dataframe['Rating'] = ratings[2]
rating_dataframe.to_csv('ratings.csv')


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(rating_dataframe, test_size = 0.3, random_state = 0)

# Preparing the training set and the test set
print("Preparing the training set and the test set\n")
training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')
train_test_dataframe_set = np.array(rating_dataframe, dtype = 'int')

# Getting the number of users and movies
print("Getting the number of users and movies\n")
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


print("Converting the data into Torch tensors\n")
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

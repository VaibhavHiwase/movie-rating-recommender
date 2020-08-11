# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import training
from collections import defaultdict 

movies = pd.read_csv('movies.csv')
movies = movies.iloc[:, 1:]
print("\n\nMovieLes dataset have "+str(training.d.nb_users)+" Users.\n\nThey gave rating to the movie after watching.\n")
print('My Recommendation System will help us to predict rating of the movies an user can give based on his/her previous'\
      ' taste (like/dislike) of the movie Genres.\n')
print("since we have total "+str(training.d.nb_users)+' users, You have to select any one UserId to run this program.')
print("Please Enter any UserId number ranging from 1 to "+str(training.d.nb_users)+'.')


while(1):
    user_id = int(input("Enter UserID: "))
    if(user_id < 1):
        print("You entered number lesser than or equal to 0. Please Enter correct number")
        continue
    elif(user_id>training.d.nb_users):
        print('Please enter number greater than 1 and less than '+str(training.d.nb_users+1))
        continue
    else:
        break
user_id = user_id - 1
###########################################################################################################
  
org=[]
cal=[]
# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(training.d.nb_users):
    input_ = Variable(training.d.training_set[id_user]).unsqueeze(0)
    target = Variable(training.d.test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = training.sae(input_)
        output = output[0]
        target.require_grad = False
        output[target == 0] = 0
        original = target[target > 0]
        org.append(original)
        calculated = output[target > 0]
        calculated = calculated.round()
        cal.append(calculated)
        loss = training.criterion(output, target)
        mean_corrector = training.d.nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
#        print(test_loss/s)
        s += 1.
print('test loss: '+str(test_loss/s))

#################################################### Training ###############################################

original_dataframe = pd.DataFrame(index = range(training.d.nb_users), columns = range(training.d.nb_movies))
 
original_rating_tensor = Variable(training.d.training_set[user_id]).unsqueeze(0)
original_rating = []
for i in range(len(original_rating_tensor[0])):
    original_rating.append(int(list(original_rating_tensor)[0][i]))
original_dataframe.iloc[user_id] = np.array(original_rating)

movies_rating = defaultdict(list)
for i in range(len(original_dataframe.iloc[user_id])):
    movies_rating[i].append(original_dataframe.iloc[user_id][i])

user_movies_rating = pd.DataFrame()
user_title = []
user_rating = []
for j in range(len(movies)):
    if(movies['MovieID'][j] in  range(1, len(movies_rating.keys())+1)):
        user_title.append(movies['Title'][j])
        user_rating.append(movies_rating[movies['MovieID'][j]-1][0])

user_movies_rating['Title'] = user_title
user_movies_rating['Rating'] = user_rating
user_movies_rating.to_csv('user_movie_rating.csv')
user_movies_rating = pd.read_csv('user_movie_rating.csv')
user_movies_rating_dd = defaultdict(list)

for i in range(len(user_movies_rating)):
    temp = list(user_movies_rating['Unnamed: 0'])[i]
    user_movies_rating_dd[temp].append(user_movies_rating['Rating'][i])  

u_index = []
u_rating = []
for i in user_movies_rating_dd.keys():
    if(user_movies_rating_dd[i][0] > 0):
        u_index.append(i + 1)
        u_rating.append(user_movies_rating_dd[i][0])

u_title = []
for i in user_movies_rating.index:
    if(i in u_index):
        u_title.append(list(user_movies_rating['Title'])[i])

user_movie_rating_training = pd.DataFrame()
user_movie_rating_training['MovieID']= u_index
user_movie_rating_training['Title']= u_title
user_movie_rating_training['Rating']= u_rating
###########################################################################################################
print('\nI used Stacked AutoEncoder deep learning model for training and prediction of the Recommendation System.')
print('\nFollowing are the MovieID and Rating used for training the model for UserID '+str(user_id+1)+' :\n\n'+str(user_movie_rating_training))
print('\n\nThus for UserID: '+str(user_id+1)+', Training was done on model using '+str(np.shape(user_movie_rating_training)[0])+' MovieID and their Rating.')
print('\n\nSimilar training was done for all other UserIDs and finally model is trained.')

#################################################### Testing ###############################################

testing_dataframe = pd.DataFrame(index = range(training.d.nb_users), columns = range(training.d.nb_movies))
test_rating_tensor = Variable(training.d.test_set[user_id]).unsqueeze(0)
test_rating = []
for i in range(len(test_rating_tensor[0])):
    test_rating.append(int(list(test_rating_tensor)[0][i]))
testing_dataframe.iloc[user_id] = np.array(test_rating)

predict_rating = []
for i in range(len(cal[user_id])):
    predict_rating.append(int(list(cal[user_id])[i]))


movies_rating = defaultdict(list)
for i in range(len(testing_dataframe.iloc[user_id])):
    movies_rating[i].append(testing_dataframe.iloc[user_id][i])

test_user_movies_rating = pd.DataFrame()
test_user_title = []
test_user_rating = []
for j in range(len(movies)):
    if(movies['MovieID'][j] in  range(1, len(movies_rating.keys())+1)):
        test_user_title.append(movies['Title'][j])
        test_user_rating.append(movies_rating[movies['MovieID'][j]-1][0])

test_user_movies_rating['Title'] = test_user_title
test_user_movies_rating['Rating'] = test_user_rating
test_user_movies_rating.to_csv('user_movie_rating.csv')
test_user_movies_rating = pd.read_csv('user_movie_rating.csv')
test_user_movies_rating_dd = defaultdict(list)

for i in range(len(test_user_movies_rating)):
    temp = list(test_user_movies_rating['Unnamed: 0'])[i]
    test_user_movies_rating_dd[temp].append(test_user_movies_rating['Rating'][i])  

tu_index = []
tu_rating = []
for i in test_user_movies_rating_dd.keys():
    if(test_user_movies_rating_dd[i][0] > 0):
        tu_index.append(i + 1)
        tu_rating.append(test_user_movies_rating_dd[i][0])

tu_title = []
for i in test_user_movies_rating.index:
    if(i in tu_index):
        tu_title.append(list(test_user_movies_rating['Title'])[i])

user_movie_rating_testing = pd.DataFrame()
user_movie_rating_testing['MovieID']= tu_index
user_movie_rating_testing['Title']= tu_title
user_movie_rating_testing['Original Rating']= tu_rating
user_movie_rating_testing['Predicted Rating'] = predict_rating
user_movie_rating_testing.to_csv('Prediction.csv')

###########################################################################################################
# Precision Recall and accuracy
# Making the Confusion Matrix
y_pred =[]
y_test =[]
user_movie_rating_testing = pd.read_csv('Prediction.csv')
del user_movie_rating_testing['Unnamed: 0']
y_test = list(user_movie_rating_testing['Original Rating'])
y_pred = list(user_movie_rating_testing['Predicted Rating'])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy=0
s=0
for i in range(len(cm)):
    for j in range(len(cm[i])):
        if(i==j):
            accuracy += cm[i][j]
        s += cm[i][j]
accuracy = (accuracy/s) * 100

loss = training.criterion(cal[user_id], org[user_id])

###########################################################################################################

print('\nUserID number '+str(user_id+1)+' provide total of '+str(np.shape(user_movie_rating_training)[0]+np.shape(user_movie_rating_testing)[0])+' ratings'\
      ' of ' +str(np.shape(user_movie_rating_training)[0]+np.shape(user_movie_rating_testing)[0])+' movies.')

print('\nOut of these '+str(np.shape(user_movie_rating_training)[0]+np.shape(user_movie_rating_testing)[0])+' ratings'\
      ' of ' +str(np.shape(user_movie_rating_training)[0]+np.shape(user_movie_rating_testing)[0])+' movies,\n'\
      +str(np.shape(user_movie_rating_training)[0])+' ratings was used for training the model and based on the training'\
      ' of the model I made prediction on the remaining '+str(np.shape(user_movie_rating_testing)[0])+ ' ratings of the '\
      +str(np.shape(user_movie_rating_testing)[0])+' movies.\n')
print('For UserId: '+str(user_id+1)+', we got accuracy of ' + str(int(accuracy))+'% for our prediction with the loss of '+str(float(loss)))
print('The confusion matrix of our prediction is as follows:\n',cm)
print('Following are the MovieID, Original Rating and Predicted Rating used for testing the model for UserID: '+str(user_id+1)+' :\n\n'+str(user_movie_rating_testing))
print('\nPlease open Prediction.csv file generated in working directory for better comparision of the results!'\
      'Naturally We can increase accuracy of our model with training by 200 epoch.')


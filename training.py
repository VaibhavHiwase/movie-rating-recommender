# AutoEncoders

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from autoencoder import SAE
import dataset as d


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

#users = pd.read_csv('users.csv')
#users = users.iloc[:, 1:]
#movies = pd.read_csv('movies.csv')
#movies = movies.iloc[:, 1:]

print("Wait for epoch comes to 0. Model is training.\nIdeally we need to increase epoch upto 200 to better train"\
      " the model. But It takes a lot of time for training. Thus I used 10 epoch for training the model."\
      " Please increase epoch for better training in future.\n")
# Training the SAE
nb_epoch = 140
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(d.nb_users):
        input = Variable(d.training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = d.nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
    print('Remaining epoch: '+str(nb_epoch - epoch)+' Training loss: '+str(train_loss/s))


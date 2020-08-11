# Movie Rating Recommendation System
Following are the steps to execute this project:

Step 1: Download Anaconda software from https://www.anaconda.com/download/ and install.

Step 2: Go to anaconda prompt and type folowing commands:

	conda install -c anaconda numpy 
	conda install -c anaconda pandas 
	conda install -c peterjc123 pytorch 

Step 3: Open Spyder IDE from anaconda prompt by typing "spyder".

Step 4: Open python code from "run.py" using file explorer.

Step 4: Set console working directory from spyder IDE.

Step 5: Run the python program. or type "import testing" on console.

Step 6: Analyze result from path "Prediction.csv" file.

The Recommendation system was build for movie rating prediction. I did not find relevant dataset for course recommendation system like udemy.com, so I decided to work on movie reommendation system. The data I used is called MovieLens from https://movielens.org/. You can read about this dataset on http://files.grouplens.org/datasets/movielens/ml-20m-README.html. This project didn't give recommendation directly as this decision is very crutial. Hence I made a movie rating recommendation system where we can predict what rating an user can give to the movie if he/she will watch that movie? This decision was taken from the tase of user about the movie like his/her likes and dislikes, Genres, etc.

I used Stacked AutoEncoder for making the prediction of movie ratings. If you are intreasted to read more about this deep learning tehchnology, you can refer to the following links and pdf attached pdfs for readings:

https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/

https://blog.keras.io/building-autoencoders-in-keras.html

http://mccormickml.com/2014/05/30/deep-learning-tutorial-sparse-autoencoder/

http://www.ericlwilkinson.com/blog/2014/11/19/deep-learning-sparse-autoencoders

Please Run run.py file. All other files will execute automatically. Recommended RAM is 8GB or more.

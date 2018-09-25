#Kaggle Competition - New York Taxi Fare Prediction Challenge

This assignment focused on working individual on a Kaggle Competition on a challenge: New York Taxi Fare Prediction. 

We were given a input data set of 55 Million tuples in order to predict the final fare amount for the cab ride in NYC.

The input features were, pickup and dropoff Latitude and Longitude, along with the time and number of passengers. 

The challenge here was the data cleaning part, since there is a huge data set, we need to remove all the outliers and garbage values that would prove harmful for the data in the future. 

Once data cleansing was done, we had to do training our model. I used two techniques: 
1. Linear Regression: I took 3 features - distance, time and passenger count in order make the prediction. For the training data set, I was getting a RMS of 3.91. But the submission on Kaggle generated a RMS Value of 5.6. So that's the reason why we had to move to some other techniques.

2. Random Forest : Same as before, we used the same features as above, and I got the best RMS value of 4.17 on Kaggle's dataset. I had tweaked a few parameters in Random Forest Regressor, like nestimators, random_state, and depth and that's when for a depth value of 12 and n_estimators value of 5, I could get a great score of 4.17

I tried other techniques as well, but those didn't work out very well, so they had to be dropped, like Neural Network, SVC but they were not able to get proper result(because of the huge size of dataset)


My Kaggle Standing: 1000 Rank in the global competition for NYC Taxi Fare Prediction

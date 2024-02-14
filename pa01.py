# EE532L - Deep Learning for Healthcare - Programming Assignment 01
# Authors: Jibitesh Saha, Sasidhar Alavala, Subrahmanyam Gorthi
# Important: Please do not change/rename the existing function names and write your code only in the place where you are asked to do it.


########################################################## Can be modified ##############################################################
# You can import libraries as per your need
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Write your code for the logistic regression below so that it returns y_test_pred
def regress_fit(X_train, y_train, X_test):
    
    # Q1) Normalize the training/testing features (Hint:- There are 8 features, so each of 
    # them should be divided by the largest value of that specific feature ,among all the 
    # samples)
    # (Note:- Take care of the axis or else accuracy won't increase)
    
    p = X_train.shape[0] # Number of features
    N = X_train.shape[1] # Number of sample cases
    
    # Q2) Set value of learning rate e
    e =  # Learning Rate (Hint:- adjust between 1e-5 to 1e-2)
    
    w = np.random.uniform(-1/np.sqrt(p), 1/np.sqrt(p), (p+1,1)) # Random initialization of
    # weights
    X = np.ones((p+1,N)) # Adding an extra column of ones to adjust biases
    X[:p,:] = X_train
   
    
    # Q3) Set number of epochs
    num_epochs = 
    for epoch in range(num_epochs): # Loop for iterative process
        J = 0 # Initializing loss
        count = 0 # Initializing count of correct predictions
        for i in range (N):
            z = ((w.T)@X[:,i:i+1])[0,0] # Raw logits (W.T x X)    
            
            # Q4) Write equation of Sigmoid(z)
            y =  # Sigmoid activation function
            T = y_train[i] # Ground Truth
            
            # Q5) Write loss function after the minus sign
            J = J- # Loss function
            # (Note:- The loss function is written after J = J- because we are trying to find
            # the average loss per epoch, so we need to sum it iteratively )
            
            # Q6) Write Derivative of J w.r.t z
            k =  # Derivative of J w.r.t z (Chain rule, J w.r.t y multiplied by y w.r.t z )
            dJ = k*X[:,i:i+1] # Final Derivative of J w.r.t w (dJ/dz multiplied by dz/dw)
            
            # Q7) Write formula of Gradient Descent
            w =  # Gradient Descent
            
            if abs(y-T)<0.5:
                count = count+1 # Counting the number of correct predictions
                
            # Q8) So the two lines of code above is a method to find TP (True positives).
            # So similarly write a code for finding TN, FP, FN
            
            
        
        
        
        train_loss = J/N
        train_accuracy = 100*count/N
        
        # Q9) Find the precision, recall, specificity, F1 score and IoU 
        

        batch_metrics = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} "
        sys.stdout.write('\r' + batch_metrics)
        sys.stdout.flush()
    
    # Q10) store all the metrics you have found above w.r.t epoch and plot the loss vs epoch
    # curve, accuracy vs epoch curve and similarly every metric you have found vs epoch curve
    # separately
    
    
    
    # Testing
    print("\n")
    N2 = X_test.shape[1] # Number of test samples

    X2 = np.ones((p+1,N2)) # adding an additional columns of 1 to adjust biases
    X2[:p,:] = X_test

    z2 = w.T@X2 # test logit matrix
    y_pred = 1/(1+np.exp(-z2)) # Sigmoid activation function to convert into probabilities
    y_pred[y_pred>=0.5] = 1 # Thresholding
    y_pred[y_pred<0.5] = 0

    return y_pred

###########################################################################################################################################
# BONUS QUESTION (For people who opted out of the project, Everyone is allowed to attempt anyway)

# Repeat this whole process for combinations of sigmoid/tanh activation function 
# and Binary Crossentropy/MSE loss function (3 more possible combinations) and compare their metric curves
############################################################################################################################################


########################################################## Cannot be modified ##############################################################
# Logistic Regression using sklearn
def regress_fit_sklearn(X_train, y_train, X_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=3)
    X_train_scaled = poly.fit_transform(X_train_scaled)
    X_test_scaled = poly.transform(X_test_scaled)

    model = LogisticRegression(max_iter=1000, C=0.5)
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)

    return y_test_pred

# Load the dataset
def load_and_fit():

    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    X2 = np.array(X)
    X2 = X2.T
    y = df["Outcome"]
    X_train = X2[:,:614]
    
    X_test = X2[:,614:]
    y_train = y[:614]
    
    y_test = y[614:]

    # Fit the model
    y_test_pred_sk = regress_fit_sklearn(X_train, y_train, X_test)
    y_test_pred = regress_fit(X_train, y_train, X_test)

    # Evaluate the accuracy
    test_accuracy_sk = accuracy_score(y_test, y_test_pred_sk)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy using sklearn: {test_accuracy_sk:.5f}")
    print(f"Test Accuracy using your implementation: {test_accuracy:.5f}")
    return round(test_accuracy, 5)

load_and_fit()
###########################################################################################################################################

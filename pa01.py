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
import matplotlib.pyplot as plt

# Write your code for the logistic regression below so that it returns y_test_pred
def regress_fit(X_train, y_train, X_test):

    # Q1) Normalize the training/testing features (Hint:- There are 8 features, so each of 
    # them should be divided by the largest value of that specific feature ,among all the 
    # samples)
    # (Note:- Take care of the axis or else accuracy won't increase)
    # Calculate mean and standard deviation for each feature in X_train
    X_train_mean = np.mean(X_train, axis=1, keepdims=True)
    X_train_std = np.std(X_train, axis=1, keepdims=True)

# Normalize X_train
    X_train = (X_train - X_train_mean) / X_train_std
    
# Calculate mean and standard deviation for each feature in X_test
    X_test_mean = np.mean(X_test, axis=1, keepdims=True)
    X_test_std = np.std(X_test, axis=1, keepdims=True)

# Normalize X_test
    X_test = (X_test - X_test_mean) / X_test_std

    p = X_train.shape[0] # Number of features
    N = X_train.shape[1] # Number of sample cases
    
    # Q2) Set value of learning rate e
    e =  0.0001  # Learning Rate (Hint:- adjust between 1e-5 to 1e-2)
    
    w = np.random.uniform(-1/np.sqrt(p), 1/np.sqrt(p), (p+1,1)) # Random initialization of
    # weights
    X = np.ones((p+1,N)) # Adding an extra column of ones to adjust biases
    X[:p,:] = X_train

    losses = []
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    ious = []

    TP,TN,FP,FN = 0,0,0,0
    # Q3) Set number of epochs
    num_epochs = 1500
    for epoch in range(num_epochs): # Loop for iterative process
        J = 0 # Initializing loss
        count = 0 # Initializing count of correct predictions
        for i in range (N):
            
            z = ((w.T)@X[:,i:i+1])[0,0] # Raw logits (W.T x X)    
            
            # Q4) Write equation of Sigmoid(z)
            y = 1/(1 + np.exp(-z)) # Sigmoid activation function
            
            T = y_train[i] # Ground Truth
           
            # Q5) Write loss function after the minus sign
            J = J-(T * np.log(y) + (1 - T) * np.log(1 - y)) # Loss function    
            # (Note:- The loss function is written after J = J- because we are trying to find
            # the average loss per epoch, so we need to sum it iteratively )
            
            # Q6) Write Derivative of J w.r.t z
            k = (((1-T)/(1-y))-(T/y))*y*(1-y) # Derivative of J w.r.t z (Chain rule, J w.r.t y multiplied by y w.r.t z )
            dJ = k*X[:,i:i+1] # Final Derivative of J w.r.t w (dJ/dz multiplied by dz/dw)
            
            # Q7) Write formula of Gradient Descent
            w = w - e * dJ # Gradient Descent
            
            if abs(y-T)<0.5:
                count = count+1 # Counting the number of correct predictions
                
            # Q8) So the two lines of code above is a method to find TP (True positives).
            # So similarly write a code for finding TN, FP, FN
            
            if y >= 0.5 and T == 1:
                TP += 1
            elif y < 0.5 and T == 0:
                TN += 1
            elif y >= 0.5 and T == 0:
                FP += 1
            elif y < 0.5 and T == 1:
                FN += 1

        train_loss = J/N
        train_accuracy = 100*count/N
        losses.append(train_loss)
        accuracies.append(train_accuracy)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        precisions.append(precision)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recalls.append(recall)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities.append(specificity)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        ious.append(iou)

    batch_metrics = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} \n"
    sys.stdout.write('\r' + batch_metrics)
    sys.stdout.flush()

    # Q9) Find the precision, recall, specificity, F1 score and IoU
    print('Learning rate: ',e)
    print("Precision:", precisions[-1])
    print("Recall:", recalls[-1])
    print("Specificity:", specificities[-1])
    print("F1 Score:", f1_scores[-1])
    print("IoU:", ious[-1])
    
    # Q10) store all the metrics you have found above w.r.t epoch and plot the loss vs epoch
    # curve, accuracy vs epoch curve and similarly every metric you have found vs epoch curve
    # separately
    
    epochs = np.arange(1, num_epochs+1)
    print(np.shape(epochs))
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')

    plt.subplot(2, 3, 2)
    plt.plot(epochs, accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')

    plt.subplot(2, 3, 3)
    plt.plot(epochs, precisions)
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision vs Epoch')

    plt.subplot(2, 3, 4)
    plt.plot(epochs, recalls)
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall vs Epoch')

    plt.subplot(2, 3, 5)
    plt.plot(epochs, specificities)
    plt.xlabel('Epoch')
    plt.ylabel('Specificity')
    plt.title('Specificity vs Epoch')

    plt.subplot(2, 3, 6)
    plt.plot(epochs, f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epoch')

    plt.tight_layout()
    plt.show()

    # Testing
    print("\n")
    N2 = X_test.shape[1] # Number of test samples

    X2 = np.ones((p+1,N2)) # adding an additional columns of 1 to adjust biases
    X2[:p,:] = X_test

    z2 = w.T@X2 # test logit matrix
    y_pred = 1/(1+np.exp(-z2)) # Sigmoid activation function to convert into probabilities
    y_pred[y_pred>=0.5] = 1 # Thresholding
    y_pred[y_pred<0.5] = 0

    return y_pred.reshape(-1)

###########################################################################################################################################
# BONUS QUESTION (For people who opted out of the project, Everyone is allowed to attempt anyway)

# Repeat this whole process for combinations of sigmoid/tanh activation function 
# and Binary Crossentropy/MSE loss function (3 more possible combinations) and compare their metric curves
############################################################################################################################################


########################################################## Cannot be modified ##############################################################
# Logistic Regression using sklearn
def regress_fit_sklearn(X_train, y_train, X_test):

    X_train = X_train.T
    X_test = X_test.T

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=3)
    X_train_scaled = poly.fit_transform(X_train_scaled)
    X_test_scaled = poly.transform(X_test_scaled)

    model = LogisticRegression(max_iter=3000, C=0.6)
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

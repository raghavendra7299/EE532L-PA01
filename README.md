# EE532L Deep Learning for Healthcare - Programming Assignment 01

## Report
Accuracy: 

Methods used to improve accuracy:

Bonus question:
![fig](assets/logo.png)
 - Observations:

## About
The Pima Indians Diabetes Database is a widely used dataset in machine learning, particularly for binary classification tasks related to diabetes prediction. The dataset consists of 768 instances. There are 8 numeric predictive attributes. The attributes are as follows:
 - Pregnancies: Number of times pregnant
 - Glucose: Plasma glucose concentration 2 hours in an oral glucose tolerance test
 - BloodPressure: Diastolic blood pressure (mm Hg)
 - SkinThickness: Triceps skin fold thickness (mm)
 - Insulin: 2-Hour serum insulin (mu U/ml)
 - BMI: Body mass index (weight in kg/(height in m)^2)
 - DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores the likelihood of diabetes based on family history)
 - Age: Age in years

The target variable is a binary variable indicating whether a patient has diabetes or not. It takes the values 0 (no diabetes) or 1 (diabetes). Now your goal is to build a predictive model using logistic regression to accurately classify individuals as diabetic or non-diabetic based on the given attributes.

## Objective

1: Please pay attention to the comments of the Python file pao1.py and rough.ipynb. There are 10 questions designed for you. Please fill in the code snippets wherever necessary. 

2: Our trusty model (based on sklearn's LogisticRegression()) is stuck at an OK-ish 0.812 accuracy and the built-from-scratch is stuck at a 0.779 accuracy – not bad, but we're aiming for the stars, not just the clouds. So find some novel way to do pre-processing or use some novel loss function to improve the accuracy.

Bonus question: The bonus question is mentioned at the bottom of the file. Whoever has opted not to do the course project should do these questions. Others are allowed to attempt as well.

No cheating, the TAs are like hawks, they see everything! May the data odds be ever in your favour!


## Instructions
  - Make sure you have a GitHub account. If you don't have one, create an account at [GitHub](https://github.com/).
  - Please accept this invite (shared in the Google Classroom) for the GitHub Classroom.
  - Once you accept the assignment invitation, you will be redirected to your assignment repository on GitHub.
  - Open GitHub codespaces and then you can use rough.ipynb to build your rough code and then make similar changes in pa01.py and commit those changes.
  - You're supposed to only change the sections where you are allowed to do so in the pa01.py script.
  - Then upload or commit and push the changes to your assignment repo.
  - Your assignment will be automatically graded, otherwise, you will receive an email that the auto-grading has failed then make sure the code has no errors.

## References
- Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

## License and Acknowledgement
The dataset is from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data?select=diabetes.csv). Please follow their licenses. Thanks for their awesome work.


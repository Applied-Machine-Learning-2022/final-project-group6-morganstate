[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8127866&assignment_repo_type=AssignmentRepo)
<!--
Name of your teams' final project
-->
# final-project
## [National Action Council for Minorities in Engineering(NACME)](https://www.nacme.org) Google Applied Machine Learning Intensive (AMLI) at the `MORGAN STATE UNIVERSITY`

<!--
List all of the members who developed the project and
link to each members respective GitHub profile
-->
Developed by: 
- [Moja Williams](https://github.com/Moja-afk) - `Morgan State University`
- [Travis Jones](https://github.com/TravisJones25) - `Morgan State University` 
- [Teqwon Norman](https://github.com/Teqwon-Norman) - `Morgan State University` 
- [Emmanuel Lewis](https://github.com/Emlew6) - `Morgan State University`

## Description
<!--
Give a short description on what your project accomplishes and what tools is uses. In addition, you can drop screenshots directly into your README file to add them to your README. Take these from your presentations.
-->
User's inboxs are bombarded with spam emails which overpopulate the inbox. In turn it makes it harder for the user to find important emails and attend to them in a timely fashion .Deciphering between legitimate emails and spam is an inherent issue which could lead users  missing important information. With in this project we will develop several models (classifiers) to remedy this issue.


## Usage instructions
<!--
Give details on how to install fork and install your project. You can get all of the python dependencies for your project by typing `pip3 freeze requirements.txt` on the system that runs your project. Add the generated `requirements.txt` to this repo.
-->
1. Fork this repo
2. Change directories into your project
3. Go to 'spam and ham ' colab file
4. Download 'spam and ham.csv'



# Capstone
ML Classifier comparison on "Spam" email Dataset


Capstone Project Overview

Project title:
	Testing the efficacy of Neural Networks models on the detection of spam
Goals:
The project goal is to construct a model that is able to compare the performance of several classifiers accurately based on both accuracy and F1 score.
To compare between ‘spam’ or ‘ham’(not spam) in an email inbox.

Intermediate goals:
To create a model the is appropriate for the data we are given
To reduce the amount of error that may arise due to (bias)
To complete the overall project by the deadline.
Data Acquisition:
Download data set
Convert data in to a readable format
Exploratory data analysis(repair data,find missing values, etc.)
Define targets and column(s) to which to aim for 
Define features columns that would be based on certain columns from the data
Train-test-split the data(30% train, 70%)
Create the model we will use to classify 
Fit model to training data
Get predictions for test values
Lasty compare all resulting outcomes.
Project Roles 
	Emmanuel Lewis: 
Researcher and logistical support
Travis Jones:
	Code and presenter
Teqwon Norman
	Lead coder and Designer
Moja
	Auditor and second coder 

Background information:
	*SVM or support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.

Example of code: 
	from  sklearn.svm import SVC
	clf = SVC( kernel = ‘linear’)
	clf .fit(X,Y)
# prediction
	prediction=clf.prediction([ [ ] ] )

Website:https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989
 
	*Logistic regression Classifier:
		Logistic Regression Classifier is a technique used in machine learning. It is used as a logistic function to model the dependent variable. The dependent variable is dichotomous in nature , there could only be two possible classes (yes or no)As a result , this technique is used while dealing with binary data.
 
Example:
	#Import the Libraries and read the data into a Pandas DataFrame
	Import pandas as pd
	Import numpy as np
	
	df= pdmread_csv(‘ insert file name’)
	df.head()
	
	#clean the data and remove missing data
		series= pd.isnull(df[‘ insert target name’])
Website:https://towardsdatascience.com/the-perfect-recipe-for-classification-using-logistic-regression-f8648e267592#:~:text=Logistic%20Regression%20is%20a%20classification%20technique%20used%20in%20machine%20learning,cancer%20is%20malignant%20or%20not).

	
*Decision Tree Classifier
A decision tree is a supervised machine learning algorithm that uses a set of rules to make decisions, similarly to how humans make decisions.
------------------
Logic Breakdown
Exploratory stage:
Load data into colab
dataset(name)= pd.read_csv(‘spam_ham.csv)
Describe data look for (labels,rows and columns,missing data)
dataset.describe()
Check for data shape and type
dataset.shape()
Check for type of data ( integer, words, figures)
Check for data flow
Plot data
histPlotAll(dataset)
boxPlotAll(dataset)
Cleaning data:
Remove duplicate or irrelevant data
Fix structural errors
Filter unwanted data outliers (streamline data)
Handle missing data
Drop missing data points
Change the value (of missing data points to represent something)
Validation
Questions to ask after cleaning.
Does the data make sense?
Does the data follow a pattern or concept
Is the data quality appropriate for the required application .
Theory behind (Why it is important standardized )
Consistency : This helps with having the same information across all programming .
Uniformity : This helps with making sure that all the variables and symbols are the same throughout the data.
Conversion : If the is presented 
Feature Scaling ( Normalization)
Create a scale for data (Min Max Scaler)
Implementing feature Normalization
Satandardize data to allow for scaling
Creating / using a preprocessor for sklearn
Preprocessing data (model.1)
Split data into train(70%) and test (30%)
Check for missing values within data 
df.isna().sum()
Convert data back to dataframe 
Convert features to integers ( check labels of columbs)
Website:https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html
Create a model (sklearn(decision))
Example: model=DecisionTreeClassifier()
#Select algorithm
model.fit(x_train,y_train)
#Fit model  to the data
Predictions = model.predict(X_train)
print(accuracy_score(y_train, predictions))
#Check model performance on training data
https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html

Preproceeing data (model.2)
support Vector Machine is a linear model
https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
Preproceeing data (model.3)
Logistic Regression Classifier 
# Create an instance of the scalar and apply it to the data
sc= StandardScalar()
X_train = sc.fit_transform(X_train)
X_test= sc.treansform(x_test)
From sklearn.linear_model import LogisticRegression
#Create classifier
LogReg_clf= LogisticRegression(random_state=42)
classifier.fit(X_train,y_train)
Code Import
#Data manipulation and visualization
Import pandas as pd
Import numpy as np
Import matplotib.pyplot as pyplot
# Machine learning libraries needed
From sklearn.model_selection import train_test_split
From sklearn.tree import DecisionTreeClassifier
From sklearn.metrics import accuracy_score

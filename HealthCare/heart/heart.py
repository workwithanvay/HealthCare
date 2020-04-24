#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing dataset
dataset=pd.read_csv('heart.csv')
x=dataset.iloc[:, 0:13].values
y=dataset.iloc[:, 13].values
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#splitting into test and training dataset
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
correct=cm[0][0]+cm[1][1]
wrong=cm[0][1]+cm[1][0]
acc=100*correct/(wrong+correct)
print("accuracy of random forest is %d"%acc)


from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', random_state = 2)
classifier1.fit(x_train, y_train)

# Predicting the Test set results
y_pred1 = classifier1.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
correct1=cm1[0][0]+cm1[1][1]
wrong1=cm1[0][1]+cm1[1][0]
acc1=100*correct1/(wrong1+correct1)
print("accuracy of SVM is %d"%acc1)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()
classifier2.fit(x_train, y_train)

# Predicting the Test set results
y_pred2= classifier2.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)
correct2=cm2[0][0]+cm2[1][1]
wrong2=cm2[0][1]+cm2[1][0]
acc2=100*correct2/(wrong2+correct2)
print("accuracy of Naive Bayes is %d"%acc2)

if(acc>acc1 and acc1>acc2):
    print("random forest with accuracy:- %d"%acc)
elif(acc1>acc2):
    print("SVM with accuracy:- %d"%acc1)
else:
    print("Naive Bayes with accuracy:- %d"%acc2)


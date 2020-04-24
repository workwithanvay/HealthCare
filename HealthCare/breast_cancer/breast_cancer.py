#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('data.csv')
x=dataset.iloc[:, 2:32].values
y=dataset.iloc[:, 1:2].values

#ploting the dataset
plt.hist(dataset['diagnosis'])
plt.title('Diagnosis (Malignant=1 , Benign=0)')
plt.show()

#encoding categorical data Malignant-1 Benign-0
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x=LabelEncoder()
y[:,0]=labelencoder_x.fit_transform(y[:,0])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

y = np.array(ct.fit_transform(y), dtype=np.float)
#Avoiding dummy variable trap
y=y[:, 1:]
y=y.reshape(569,)


#splitting into test and training dataset
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, min_samples_split=25, max_depth=7,criterion = 'entropy', random_state = 0)
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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix 

ds = pd.read_csv("C:\\Users\\kvsr\\Desktop\\ECE 760 ML\\hw3-1\\data\\emails.csv")
results = []
for i in range(5):
    x_test = ds.iloc[1000*i:1000*(i+1),1:3001]
    y_test = ds.iloc[1000*i:1000*(i+1),3001]
    df_train = ds.drop(ds.index[range(1000*i,1000*(i+1))])
    x_train = df_train.iloc[:,1:3001]
    y_train = df_train.iloc[:,3001]
    #st_x= StandardScaler()    
    #x_train= st_x.fit_transform(x_train)    
    #x_test= st_x.transform(x_test)
    classifier= KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm= confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    #accuracy_sum = accuracy_sum + accuracy
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("fold %d\n"%(i+1))
    print("accuracy = %f\n"%(accuracy))
    print("precision = %f\n"%(precision))
    print("recall = %f\n"%(recall))
    print("\n")
    results.append([accuracy,precision,recall])
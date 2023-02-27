import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt

ds = pd.read_csv("C:\\Users\\kvsr\\Desktop\\ECE 760 ML\\hw3-1\\data\\emails.csv")
results = []
for k in [1,3,5,7,10]:
    accuracy_sum = 0
    for i in range(5):
        x_test = ds.iloc[1000*i:1000*(i+1),1:3001]
        y_test = ds.iloc[1000*i:1000*(i+1),3001]
        df_train = ds.drop(ds.index[range(1000*i,1000*(i+1))])
        x_train = df_train.iloc[:,1:3001]
        y_train = df_train.iloc[:,3001]
        #st_x= StandardScaler()    
        #x_train= st_x.fit_transform(x_train)    
        #x_test= st_x.transform(x_test)
        classifier= KNeighborsClassifier(n_neighbors=k, metric='euclidean', p=2)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        cm= confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        accuracy_sum = accuracy_sum + accuracy
        #precision = precision_score(y_test, y_pred)
        #recall = recall_score(y_test, y_pred)
        #results.append([accuracy,precision,recall])
    
    average_accuracy = accuracy_sum/5
    results.append(average_accuracy)
	
k = [1,3,5,7,10]
plt.plot(k,results)
plt.xlabel("k")
plt.ylabel("average accuracy")
plt.show
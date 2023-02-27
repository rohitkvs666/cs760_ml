import numpy as np
import math
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1/(1+ np.exp(-x))

class LogisticRegression():
    
    def __init__(self,learning_rate = 0.01, n=150):
        self.learning_rate = learning_rate
        self.n = n
        self.weights = None
        self.bias = None
    
    def fit(self,x,y):
        
        num_training_data, num_features = x.shape
        #num_features = np.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for i in range(self.n):
            y_linear = np.dot(x, self.weights) + self.bias
            y_pred = sigmoid(y_linear)
            #dw = np.mean(np.dot(x.T, (y_pred - y)))
            #db = np.mean(y_pred - y)
            dw = (1/num_training_data)*(np.dot(x.T, (y_pred - y)))
            db = (1/num_training_data)*(np.sum(y_pred-y))
            
            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db
        
        return self.weights, self.bias
    
    def predict(self, x_test, threshold=0.5):
        y_prob = sigmoid(np.dot(x_test, self.weights) + self.bias)
        predicted_class = []
        for k in y_prob:
            if (k > threshold):
                predicted_class.append(1)
            else:
                predicted_class.append(0)
        return predicted_class
    
    def accuracy(self,y_test,y_pred):
        count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                count=count+1
        return count/len(y_pred)
    
    def confusion_matrix(self,y_test,y_pred):
        tp = fp = fn = tn = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                if y_pred[i] == 1:
                    tp = tp+1
                else:
                    tn = tn+1
            elif y_test[i] == 0 and y_pred[i] == 1:
                fp = fp + 1
            elif y_test[i] == 1 and y_pred[i] == 0:
                fn = fn + 1
        return [tn,fp,fn,tp]
        
    

ds = pd.read_csv("C:\\Users\\kvsr\\Desktop\\ECE 760 ML\\hw3-1\\data\\emails.csv")
results = []
#lr_arr = [0.001, 0.005, 0.01, 0.05, 0.1]

for i in range(5):
    x_test_df = ds.iloc[1000*i:1000*(i+1),1:3001]
    y_test_df = ds.iloc[1000*i:1000*(i+1),3001]
    df_train = ds.drop(ds.index[range(1000*i,1000*(i+1))])
    x_train_df = df_train.iloc[:,1:3001]
    y_train_df = df_train.iloc[:,3001]
    x_test = x_test_df.to_numpy()
    y_test = y_test_df.to_numpy()
    x_train = x_train_df.to_numpy()
    y_train = y_train_df.to_numpy()
    st_x= StandardScaler()    
    x_train= st_x.fit_transform(x_train)    
    x_test= st_x.transform(x_test)
    classifier= LogisticRegression(learning_rate=0.03)

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    cm = classifier.confusion_matrix(y_test,y_pred)
    acc = (cm[0]+cm[3])/sum(cm)
    precision = cm[3]/(cm[3]+cm[1])
    recall = cm[3]/(cm[3]+cm[2])
    results.append([acc,precision,recall])

for i in range(len(results)):
    print("Fold %d"%(i+1))
    print("  Accuracy = %.3f"%(results[i][0]))
    print("  Precision = %.3f"%(results[i][1]))
    print("  Recall = %.3f"%(results[i][2]))
        
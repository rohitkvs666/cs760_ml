import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix 
from sklearn import metrics
import matplotlib.pyplot as plt


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
            
            
    
    def predict(self, x_test, threshold=0.75):
        y_prob = sigmoid(np.dot(x_test, self.weights) + self.bias)
        predicted_class = []
        for k in y_prob:
            if (k > threshold):
                predicted_class.append(1)
            else:
                predicted_class.append(0)
        return predicted_class
    def predict_prob(self, x_test):
        return (sigmoid(np.dot(x_test, self.weights) + self.bias))
    
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
x_test = ds.iloc[4000:,1:3001]
y_test = ds.iloc[4000:,3001]
x_train = ds.iloc[:4000,1:3001]
y_train = ds.iloc[:4000,3001]
classifier= KNeighborsClassifier(n_neighbors=5, metric='euclidean', p=2)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
cm= confusion_matrix(y_test, y_pred)
#print(cm)
y_test_proba = classifier.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_test_proba)

x_test_np = x_test.to_numpy()
y_test_np = y_test.to_numpy()
x_train_np = x_train.to_numpy()
y_train_np = y_train.to_numpy()
st_x= StandardScaler()    
x_train_np= st_x.fit_transform(x_train_np)    
x_test_np= st_x.transform(x_test_np)
clf_lr= LogisticRegression(learning_rate=0.03)
clf_lr.fit(x_train_np,y_train_np)
y_pred_np = clf_lr.predict(x_test_np,0.5)
y_pred_prob = clf_lr.predict_prob(x_test_np)
fpr_lr , tpr_lr , thresholds_lr = metrics.roc_curve(y_test_np,y_pred_prob)

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.plot(fpr_lr,tpr_lr)
plt.legend(["KNN", "Logistic Regression"], loc ="lower right")
plt.show

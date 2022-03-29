import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob

image = cv2.imread('a.jpeg')
image.shape
# (1109, 1616, 3) ----> (Width, Height, Encoding {RGB})
# Total no of pixels in a.jpeg = 1279 * 1618 * 3 = 6208266 

#Reading the entire folder
normal = [cv2.imread(file) for file in glob.glob('Normal_New/*.jpeg')]

covid = [cv2.imread(file) for file in glob.glob('Covid/*.jpeg')]

#EDA
cv2.imshow('first',normal[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('first',covid[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

=

covid_data = [cv2.resize(image, (300,300)) for image in covid]
normal_data = [cv2.resize(image, (300,300)) for image in normal]

cv2.imshow('first',covid_data[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('first',normal_data[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

normal_data = np.array(normal_data)
covid_data = np.array(covid_data)
covid_data.shape 


X = np.concatenate([covid_data, normal_data])



normal_label = np.zeros(len(normal_data))
covid_label = np.ones(len(covid_data))

y = np.concatenate([covid_label, normal_label])

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y)


y_train_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_train]

# Eda

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[i])
    plt.xlabel(y_train_name[i])
plt.tight_layout()
plt.show()

# Flattening the images

X_train_f = X_train.reshape(101, 270000)
X_test_f = X_test.reshape(34, 270000)

# Implementing the ML model

import lightgbm as lgb
from lightgbm import LGBMClassifier
lgb_clf = LGBMClassifier()
lgb_clf.fit(X_train_f, y_train)
lgb_clf.score(X_train_f, y_train) #100 #overfitting 
lgb_clf.score(X_test_f, y_test) #85.2 

y_pred = lgb_clf.predict(X_test_f)

confusion_matrix(y_train, y_pred_train)
confusion_matrix(y_test, y_pred_test)

y_pred_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_pred]
y_test_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_test]

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test_f[i].reshape(300, 300, 3))
    plt.xlabel('Actual Label: {}\nPrediction: {}'.format(y_test_name[i], y_pred_name[i]))
plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve


def covid_ml_module(path, dimension, estimator):
    
    normal = [cv2.imread(file) for file in glob.glob('Normal_New/*.jpeg')]
    covid = [cv2.imread(file) for file in glob.glob(path+'Covid/*.jpeg')]
        
    covid_data = [cv2.resize(image, dimension) for image in covid]
    normal_data = [cv2.resize(image, dimension) for image in normal]

    normal_data = np.array(normal_data)
    covid_data = np.array(covid_data)
    
    X = np.concatenate([covid_data, normal_data])
    normal_label = np.zeros(len(normal_data))
    covid_label = np.ones(len(covid_data))
    y = np.concatenate([covid_label, normal_label])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_train_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_train]
    
    no_of_features = dimension[0] * dimension[1] * 3
    
    X_train_f = X_train.reshape(len(y_train), no_of_features)
    X_test_f = X_test.reshape(len(y_test), no_of_features)
    
    estimator.fit(X_train_f, y_train)
    print("**************************************************")
    print("________"+estimator.__class__.__name__+"__________")
    print()
    print("Train Accuracy: {}".format(estimator.score(X_train_f, y_train)))
    print("Test Accuracy: {}".format(estimator.score(X_test_f, y_test) ))
    print()
    print("**************************************************")
    y_pred = estimator.predict(X_test_f)
    y_pred_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_pred]
    y_test_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_test]

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_test_f[i].reshape(dimension[0], dimension[1], 3))
        plt.xlabel('Actual Label: {}\nPrediction: {}'.format(y_test_name[i], y_pred_name[i]))
    plt.tight_layout()
    plt.show()

path = "C:/Users/PARUL/Downloads/"
dimension = (100, 100)
covid_ml_module(path, dimension, LGBMClassifier())























































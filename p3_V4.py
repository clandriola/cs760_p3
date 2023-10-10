#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
#import math
#import random


# # 2.1

# ### Import Data

# In[2]:


data_2_1 =  pd.read_csv(r'hw3Data/D2z.txt', sep=' ', names=["x1","x2","y"])


# ### Create Grid

# In[3]:


x1 = np.arange(-2, 2.1, 0.1)
x2 = np.arange(-2, 2.1, 0.1)
x1 = np.around(x1, decimals=1)
x2 = np.around(x2, decimals=1)


# ### Create Function for 1NN

# In[4]:


def nn1(point_array,train_data):
    distance_min = None
    prediction_final = None
    point_test = np.array(point_array)
    for index, row in train_data.iterrows():
        point_train = np.array(row)[:-1]
        distance = np.linalg.norm(point_test - point_train, 2) 
        prediction = train_data.loc[index,"y"]
        if distance_min == None:
            distance_min = distance
            prediction_final = prediction
        else:
            if distance_min > distance:
                distance_min = distance
                prediction_final = prediction
    return prediction_final


# ### Plot Grid

# In[5]:


plt.figure(figsize=(8, 6))
# Calculate 1NN for the grid
for i in x1:
    for j in x2:
        prediction = nn1([i,j],data_2_1)
        if prediction == 1:
            color = "blue"
        else:
            color = "red"
        plt.scatter(i, j, color=color, s=1)
# Plot train data
for index, row in data_2_1.iterrows():
    i = row["x1"]
    j = row["x2"]
    prediction = row["y"]
    if prediction == 1:
        color = "blue"
    else:
        color = "red"
    plt.scatter(i, j, color=color, s=4)

plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig("2_1.png")
plt.title('2D Scatter Plot')


# # 2.2

# ### Import Data

# In[6]:


data_2_2 = pd.read_csv(r'hw3Data/emails.csv')


# ### Create Train and Test Set

# In[7]:


train_single = data_2_2.loc[1:4000].reset_index(drop=True)
test_single = data_2_2.loc[4001:5000].reset_index(drop=True)


# In[8]:


fold_ranges = [(0, 999), (1000, 1999), (2000, 2999), (3000, 3999), (4000, 4999)]
fold = {}
for i, (start_idx, end_idx) in enumerate(fold_ranges):
    test_set = data_2_2.loc[start_idx:end_idx]
    list_email = list(test_set["Email No."])
    train_set = data_2_2[~data_2_2['Email No.'].isin(list_email)]
    fold[i+1] = (train_set,test_set)


# ### Create functions for accuracy, precision, and recall

# In[9]:


def metrics(predicted_list, truth_list):
    TP = 0 
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(predicted_list)):
        predicted = predicted_list[i]
        truth = truth_list[i]
        if truth == 1:
            if predicted == 1:
                TP +=1
            else:
                FN +=1
        else:
            if predicted == 1:
                FP +=1
            else:
                TN +=1       
    accuracy = (TP + TN)/(TP+FP+FN+TN)
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
    return accuracy, precision, recall 


# ### Create Function for 1NN

# In[10]:


def nn1(point_array,train_data):
    distance_min = None
    prediction_final = None
    point_test = np.array(point_array)
    for index, row in train_data.iterrows():
        point_train = np.array(row)[:-1]
        distance = np.linalg.norm(point_test - point_train, 2) 
        prediction = train_data.loc[index,"Prediction"]
        if distance_min == None:
            distance_min = distance
            prediction_final = prediction
        else:
            if distance_min > distance:
                distance_min = distance
                prediction_final = prediction
    return prediction_final


# ### Implement 1NN 5-fold Cross Validation

# In[11]:


result_2_2 = {}
for f in fold:
    train = copy.copy(fold[f][0])
    test = copy.copy(fold[f][1])
    columns_to_drop = ['Email No.']
    train = train.drop(columns=columns_to_drop)
    test = test.drop(columns=columns_to_drop)
    for i, row in test.iterrows():
        point_array = np.array(row)[:-1]
        test.loc[i,"y_predicted"] = Knn(point_array,train)
        print(i)
    predicted_list = list(test["y_predicted"])
    truth_list = list(test["Prediction"])  
    accuracy, precision, recall = metrics(predicted_list, truth_list)
    result_2_2[f] = (accuracy, precision, recall) 
    print(accuracy, precision, recall)   


# In[ ]:


result_2_2


# # 2.3

# ### Create Function for Logistic Regression

# In[ ]:


def logistic(learning_rate,train):
    columns_to_drop = ["Prediction"]
    y = np.array(train["Prediction"])
    x = np.array(train.drop(columns=columns_to_drop))
    theta = np.zeros((x.shape[1],))
    m = x.shape[0]
    print(m,x.shape,y.shape,theta.shape)
    loss_values = []
    for r in range(100000):
        if r%1000 == 0:
            print(r)
        dot_product = np.dot(x, theta)
        sigmoid = 1 / (1 + np.exp(-dot_product))
        loss = -np.mean(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid))
        loss_values.append(loss)
        grad_loss = np.dot((sigmoid - y), x)/m  
        new_theta = theta - learning_rate * grad_loss
        
        theta = new_theta
    return theta, loss_values


# ### Run 5-Fold Logistic Regression

# In[ ]:


learning_rate = 0.001
result_2_3 = {}
for f in fold:
    train = copy.copy(fold[f][0])
    test = copy.copy(fold[f][1])
    columns_to_drop = ['Email No.']
    train = train.drop(columns=columns_to_drop)
    test = test.drop(columns=columns_to_drop)
    train['Bias_collumn'] = 1
    test['Bias_collumn'] = 1
    optimal_theta, list_loss = logistic(learning_rate,train)

    truth_list = list(test["Prediction"])  
    columns_to_drop = ["Prediction"]
    x_test = np.array(test.drop(columns=columns_to_drop))
    
    prediction = 1 / (1 + np.exp(-np.dot(x_test, optimal_theta)))
    test['Result'] = prediction
    for i, row in test.iterrows():
        if row['Result']> 0.5:
            row['Result'] = 1
        else:
            row['Result'] = 0

    predicted_list = list(test["Result"])
    accuracy, precision, recall = metrics(predicted_list, truth_list)
    print(accuracy, precision, recall)
    result_2_3[f] = (accuracy, precision, recall)


# # 2.4

# ### Create Function for general KNN

# In[ ]:


def Knn(point_array, train_data, k=1):
    distances = []  
    point_test = np.array(point_array)
    for index, row in train_data.iterrows():
        point_train = np.array(row)[:-1]
        distance = np.linalg.norm(point_test - point_train, 2) 
        prediction = train_data.loc[index, "Prediction"]
        distances.append((distance, prediction))  
        
    distances.sort(key=lambda x: x[0])
    nearest_neighbors = distances[:k]
    predictions = []
    for _, prediction in nearest_neighbors:
        predictions.append(prediction)
    if sum(predictions) > k/2:
        prediction_final = 1
    else:
        prediction_final = 0
    return prediction_final


# ### Implement kNN 5-fold Cross Validation

# In[ ]:


result_2_4 = {}
for f in fold:
    result_2_4[f] = {}
    for k in [1,3,5,7,10]:
        train = copy.copy(fold[f][0])
        test = copy.copy(fold[f][1])
        columns_to_drop = ['Email No.']
        train = train.drop(columns=columns_to_drop)
        test = test.drop(columns=columns_to_drop)
        for i, row in test.iterrows():
            point_array = np.array(row)[:-1]
            test.loc[i,"y_predicted"] = Knn(point_array,train,k)
            print(i)
        predicted_list = list(test["y_predicted"])
        truth_list = list(test["Prediction"])  
        accuracy, precision, recall = metrics(predicted_list, truth_list)
        result_2_4[f][k] = (accuracy, precision, recall) 
        print(accuracy, precision, recall)   


# In[ ]:


result_2_4


# In[ ]:


average = {}
for k in [1,3,5,7,10]:
    average[k] = 0 
    for f in result_2_4:
        average[k] += result_2_4[f][k][0]
    average[k] = average[k]/5


# In[ ]:


average


# In[ ]:


x = list(average.keys())
y = list(average.values())

# Creating the line plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', color='b', label='Data Points')
plt.title('Line Plot')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("2_4.png")
plt.show()


# # 2.5 

# ### Modify KNN fuction to return metrics for the different thresholds 

# In[ ]:


def Knn_t(point_array, train_data, k=5,t=0):
    distances = []  
    point_test = np.array(point_array)
    for index, row in train_data.iterrows():
        point_train = np.array(row)[:-1]
        distance = np.linalg.norm(point_test - point_train, 2) 
        prediction = train_data.loc[index, "Prediction"]
        distances.append((distance, prediction))  
        
    distances.sort(key=lambda x: x[0])
    nearest_neighbors = distances[:k]
    predictions = []
    for _, prediction in nearest_neighbors:
        predictions.append(prediction)
        
    if sum(predictions) >= t:
        prediction_final = 1
    else:
        prediction_final = 0
    return prediction_final


# ### Add False Positive Rate to Metrics Function

# In[ ]:


def metrics_v2(predicted_list, truth_list):
    TP = 0 
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(predicted_list)):
        predicted = predicted_list[i]
        truth = truth_list[i]
        if truth == 1:
            if predicted == 1:
                TP +=1
            else:
                FN +=1
        else:
            if predicted == 1:
                FP +=1
            else:
                TN +=1       
    accuracy = (TP + TN)/(TP+FP+FN+TN)
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
    false_positive_rate = (FP)/(TN + FP)
    return accuracy, precision, recall, false_positive_rate


# ### Implement ROC curve for 5NN

# In[ ]:


result_2_5_a = {}
for t in [0,1,2,3,4,5]:
    train = copy.copy(train_single)
    test = copy.copy(test_single)
    columns_to_drop = ['Email No.']
    train = train.drop(columns=columns_to_drop)
    test = test.drop(columns=columns_to_drop)
    for i, row in test.iterrows():
        point_array = np.array(row)[:-1]
        test.loc[i,"y_predicted"] = Knn_t(point_array,train,5,t)
        print(i)
    predicted_list = list(test["y_predicted"])
    truth_list = list(test["Prediction"])  
    accuracy, precision, recall, false_positive_rate = metrics_v2(predicted_list, truth_list)
    result_2_5_a[t] = (false_positive_rate, recall)  
result_2_5_a[6] = (0,0)


# In[ ]:


result_2_5_a


# ### Implement Logistic Regression

# In[ ]:


learning_rate = 0.001
result_2_5_b = {}
train = copy.copy(train_single)
test = copy.copy(test_single)
columns_to_drop = ['Email No.']
train = train.drop(columns=columns_to_drop)
test = test.drop(columns=columns_to_drop)
train['Bias_collumn'] = 1
test['Bias_collumn'] = 1
optimal_theta, list_loss = logistic(learning_rate,train)

truth_list = list(test["Prediction"])  
columns_to_drop = ["Prediction"]
x_test = np.array(test.drop(columns=columns_to_drop))
    
prediction = 1 / (1 + np.exp(-np.dot(x_test, optimal_theta)))
test['Result'] = prediction
print(test['Result'])
for tn in range(1000):
    t = tn/1000
    test_2 = copy.copy(test)
    for i, row in test_2.iterrows():
        if row['Result']> t:
            row['Result'] = 1
        else:
            row['Result'] = 0
    predicted_list = list(test_2["Result"])
    accuracy, precision, recall, false_positive_rate = metrics_v2(predicted_list, truth_list)
    result_2_5_b[t] = (recall, false_positive_rate)


# In[ ]:


result_2_5_b


# ### Plot ROC

# In[ ]:


# KNN
plt.figure(figsize=(8, 6))
x_values = [point[0] for point in result_2_5_a.values()]
y_values = [point[1] for point in result_2_5_a.values()]
plt.plot(x_values, y_values, marker='o', linestyle='-', color='r', label='KNN')

true_positive_rates = [result_2_5_b[i]['True Positive Rate'] for i in result_2_5_b]
false_positive_rates = [result_2_5_b[i]['False Positive Rate'] for i in result_2_5_b]
plt.plot(false_positive_rates, true_positive_rates, marker='o', color='b', linestyle='-', linewidth=2, label='Logistic Regression')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig("2_5.png")
plt.show()


# In[ ]:





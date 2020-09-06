#!/usr/bin/env python
# coding: utf-8


import os
from collections import Counter
import random
import numpy as np
import pandas as pd
from scipy.stats import logistic
from numpy.linalg import inv,norm
import matplotlib.pyplot as plt
import time



def read_files(directory):
    data = {}
    for files in os.listdir(directory):
        if files != 'index.csv':
            with open(os.getcwd() + "/" + directory + "/" + files, "r") as f:
                data[int(files)] = f.read()
    return data



tic = time.time()
data = read_files('pp4data/20newsgroups')


def words(doc1):
    """
    Function to Split the document to words
    
    Parameters:
    doc1 - documents to be split
    
    Returns:
    List of words for each document
    
    """
    return[word for sent in doc1 for word in sent.split()]


unique_words = list(Counter(words(list(data.values()))).keys())

v = len(unique_words)

wn = {}
dn = {}
zn = {}
words = []
k = 20
z = 0
for key,value in data.items():
    for each in value.split():
        words.append(each)
        wn[z] = unique_words.index(each)
        dn[z] = key-1
        zn[z] = random.randint(0,k-1)
        z += 1

pi_n = np.random.permutation(len(words))

d = len(data)

cd = {j: {i: 0 for i in range(k)} for j in range(d)}
ct = {j: {i: 0 for i in range(v)} for j in range(k)}
p = np.zeros(k)


for i in range(len(words)):
    cd[dn[i]][zn[i]] += 1
    ct[zn[i]][wn[i]] += 1

# Gibbs sampler

n_iter = 500
alpha = 5/k
beta = 0.01
for i in range(n_iter):
    for n in range(len(words)):
        word = wn[pi_n[n]]
        topic = zn[pi_n[n]]
        doc = dn[pi_n[n]]
        cd[doc][topic] = cd[doc][topic] - 1
        ct[topic][word] = ct[topic][word] - 1
        for k_each in range(k):
            first = (ct[k_each][word]+beta)/((v*beta) + (sum(ct[k_each].values())))
            second = (cd[doc][k_each]+alpha)/((k*alpha) + (sum(cd[doc].values())))
            p[k_each] = first * second
        p = np.divide(p, np.sum(p))
        topic = np.random.choice(range(0,k),p = p)
        zn[pi_n[n]] = topic
        cd[doc][topic] = cd[doc][topic] + 1
        ct[topic][word] = ct[topic][word] + 1


final_ct = []
for key, value in ct.items():
    temp = []
    for key1, value1 in value.items():
        temp += [value1]
    final_ct += temp
        

final_ct1 = np.array(final_ct).reshape(k, v)


# Final Topics

topics = {}
for i in range(len(ct)):
    for j in final_ct1[i].argsort()[-5:][::-1]:
        if i in topics:
            topics[i] += [unique_words[j]]
        else:
            topics[i] = [unique_words[j]]


print(topics)


# Writing the topics

topic = []
words = []
for key,value in topics.items():
    topic.append(key+1)
    words.append(value)
    
pd.DataFrame({'topics' : topic, 'words' : words}).to_csv("topics.csv", index = None)
    


topic_rep = np.zeros((len(data), k))

for i in range(len(data)):
    for j in range(k):
        topic_rep[i][j] = (cd[i][j] + alpha)/((k*alpha) + (sum(cd[i].values())))


### Question 2

def weight_matrix(x_data, y_data, model, phi, s, alpha):
    """
    Function to calculate Wmap given the input data and corresponsing labels
    
    Parameters:
    x_data      - Independent variables
    y_data      - labels
    model       - Type of model (logistic, poisson, ordinal)
    phi         - Threshold values for ordinal regression and empty list for others
    s and alpha - Control parameter for the spread of the distribution
    
    Returns:
    Wmap for the given data and number of iterations it took to converge
    """
    #y_data = y_data.reshape(-1,1)
    w = np.zeros((x_data.shape[1],1))
    count = 0
    while True:
        a = x_data.dot(w)
        
        if model == "logistic":
            yi = logistic.cdf(a)
            d = y_data - yi
            r = yi*(1-yi)
        elif model == "poisson":
            yi = np.exp(a)
            d =  np.subtract(y_data,yi)
            r = yi
        else:
            yi = logistic.cdf(s*(phi - a))
            d = [yi[i][y_data[i]] + yi[i][y_data[i]-1]-1 for i in range(len(x_data))]
            d = np.array(d)
            r = [s**2*((yi[i][y_data[i]]*(1-yi[i][y_data[i]])) + (yi[i][y_data[i]-1]*(1-yi[i][y_data[i]-1]))) for i in range(len(x_data))]
            r = np.array(r)
            
        g = x_data.transpose().dot(d) - (alpha*w)
        r = np.diagflat(r)
        h_inv = inv(-x_data.transpose().dot(r).dot(x_data)-(alpha*np.identity(x_data.shape[1])))  
        w_new = w - h_inv.dot(g)
        if np.divide(norm(w_new-w,2),norm(w,2)) < 0.001 or count == 100:
            break
        w = w_new
        count += 1
    return w_new,count


def glm(x_data, y_data, model, phi, s, alpha):
    
    """
    Function to implement Generalized Linear Models (logistic, poisson and ordinal)
    
    Parameters:
    x_data      - Independent variables
    y_data      - labels
    model       - Type of model (logistic, poisson, ordinal)
    phi         - Threshold values for ordinal regression and empty list for others
    s and alpha - Control parameters for the spread of the distribution
    
    Returns:
    Prints the plot, number of iterations and time elapsed
    """
    y_data = y_data.reshape(-1,1)
    tic = time.time()
    x_data = np.hstack((np.ones((x_data.shape[0],1)), x_data))
    accuracy = {}
    time_w = {}
    count_final = []
    for i in range(30):
        index = random.sample(range(0, x_data.shape[0]), int(x_data.shape[0]*0.67))
        x_train = x_data[index]
        y_train = y_data[index]
        x_test = x_data[~np.isin(range(len(x_data)),index)]
        y_test = y_data[~np.isin(range(len(y_data)),index)]

        for each_fold in np.arange(0.1, 1.1, 0.1):
            #index = random.sample(range(0, x_train.shape[0]), int(x_train.shape[0]*each_fold))
            x_train_sample = x_train[range(0,int(each_fold*x_train.shape[0]))]
            y_train_sample = y_train[range(0,int(each_fold*y_train.shape[0]))]
            
            tic_w = time.time()
            w_updated, count = weight_matrix(x_train_sample, y_train_sample, model, phi, s, alpha)
            toc_w = time.time()
            count_final.append(count)
            y_hat_temp = x_test.dot(w_updated)
            
            if model == "logistic":
                y_hat = np.where(y_hat_temp >= 0, 1, 0)
                if each_fold in accuracy:
                    accuracy[each_fold].append(np.mean(y_hat == y_test))
                else:
                    accuracy[each_fold] = [np.mean(y_hat == y_test)]
                    
                if each_fold in time_w:
                    time_w[each_fold].append(toc_w-tic_w)
                else:
                    time_w[each_fold] = [toc_w-tic_w]
                    
            elif model == "poisson":
                y_hat = np.floor(np.exp(y_hat_temp))
                sum1 = 0
                for j in range(len(y_test)):
                    sum1+=abs(y_test[j]-y_hat[j])
                sum1 = sum1/len(y_test)
                if each_fold in accuracy:
                    accuracy[each_fold].append(sum1)
                else:
                    accuracy[each_fold] = [sum1]
                    
                if each_fold in time_w:
                    time_w[each_fold].append(toc_w-tic_w)
                else:
                    time_w[each_fold] = [toc_w-tic_w]
                    
            else:
                y = logistic.cdf(s*(phi - y_hat_temp))
        
                pj = np.empty((y.shape[0],1))
                for j in range(1, len(phi)):
                    pj = np.hstack((pj, (y[:,j]-y[:,j-1]).reshape(-1,1)))
                pj = pj[:,1:]

                y_hat = np.argmax(pj, axis = 1)+1

                sum1 = 0
                for j in range(len(y_test)):
                    sum1+=abs(y_test[j]-y_hat[j])
                sum1 = sum1/len(y_test)
                if each_fold in accuracy:
                    accuracy[each_fold].append(sum1)
                else:
                    accuracy[each_fold] = [sum1]
                    
                if each_fold in time_w:
                    time_w[each_fold].append(toc_w-tic_w)
                else:
                    time_w[each_fold] = [toc_w-tic_w]
                
    toc = time.time()
    
    mean_time = list(map(lambda x: np.mean(time_w[x]), time_w))
    
    mean = list(map(lambda x: np.mean(accuracy[x]), accuracy))
    sd = list(map(lambda x: np.std(accuracy[x]), accuracy))
    
    return mean,sd


y_data = np.array(pd.read_csv("pp4data/20newsgroups/index.csv", header = None))

y_data = y_data[:,1]

doc_order = list(data.keys())

doc_order = doc_order - np.ones(len(doc_order))
doc_order = [int(i) for i in doc_order]

y_data_new = y_data[doc_order]


# Mean and standard deviation of accuracy for Topic representation method
lda_mean, lda_sd = glm(topic_rep, y_data, "logistic", [], 1, 0.01)


def bag_of_words(data_list):
    m = 0
    dic = {}
    bow = []
    
    for sent in data_list:
        for word in sent.split():
            if word not in dic.keys():
                dic[word] = m
                m += 1
    for sent in data_list:
        tmp = [0 for q in range(len(dic.keys()))]
        for word in sent.split():
            tmp[dic[word]] += 1
        bow += [tmp]
    return bow



## Mean and standard deviation of accuracy for Bag of words method

bow_mean, bow_sd = glm(np.array(bag_of_words(list(data.values()))), y_data_new, "logistic", [], 1, 0.01)



# Plotting both the values

plt.figure()
plt.errorbar([i for i in np.arange(0.1, 1.1, 0.1)], lda_mean, yerr = lda_sd, label = 'LDA')
plt.errorbar([i for i in np.arange(0.1, 1.1, 0.1)], bow_mean, yerr = bow_sd, label = 'Bag of words')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.xlabel('Sub samples')

toc = time.time()
print("Time Elapsed : ", toc - tic, " Seconds")


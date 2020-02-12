
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import welch
from detect_peaks import detect_peaks
from scipy.stats import kurtosis
from scipy.stats import skew
from datetime import datetime


# In[2]:


#Import Classifiers
#https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
#Linear Models
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Hinge
from sklearn.linear_model import Huber
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LarsCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Log
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import ModifiedHuber
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RANSACRegressor
#from sklearn.linear_model import RandomizedLasso
#from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SquaredLoss
from sklearn.linear_model import TheilSenRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Linear and Quadratic Discriminant Analysis
from sklearn.discriminant_analysis import BaseEstimator
from sklearn.discriminant_analysis import ClassifierMixin
from sklearn.discriminant_analysis import LinearClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import StandardScaler
from sklearn.discriminant_analysis import TransformerMixin

#Kernel Ridge Regression
from sklearn.kernel_ridge import BaseEstimator
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_ridge import RegressorMixin

#Support Vector Machines
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVC
from sklearn.svm import NuSVR
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.svm import SVR

#Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor

#Nearest Neighbors
from sklearn.neighbors import BallTree
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KernelDensity
#from sklearn.neighbors import LSHForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor

#Gaussian Processes
#from sklearn.gaussian_process import GaussianProcess
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier

#Cross Decomposition
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import PLSSVD


#Naive Bayes
from sklearn.naive_bayes import ABCMeta
from sklearn.naive_bayes import BaseDiscreteNB
from sklearn.naive_bayes import BaseEstimator
from sklearn.naive_bayes import BaseNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import LabelBinarizer
from sklearn.naive_bayes import MultinomialNB

#Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier

#Ensemble Methods
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import VotingClassifier


#Multiclass and multilabel algorithms
from sklearn.multiclass import BaseEstimator
from sklearn.multiclass import ClassifierMixin
from sklearn.multiclass import LabelBinarizer
from sklearn.multiclass import MetaEstimatorMixin
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import Parallel
from sklearn.multioutput import ABCMeta
from sklearn.multioutput import BaseEstimator
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import ClassifierMixin
from sklearn.multioutput import MetaEstimatorMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputEstimator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import Parallel
from sklearn.multioutput import RegressorMixin

#Semi-Suportvised
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import label_propagation

#Isotonic Regression
from sklearn.isotonic import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.isotonic import RegressorMixin
from sklearn.isotonic import TransformerMixin

# Neural network models (supervised)
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


# In[3]:


from functions import UserInputs2WorkingForm
from functions import GetTrainingData
from functions import GetTrainingData2
from functions import GetAllModelsForComparison
from functions import GetTESTDataFrameNames
from functions import getBarPlot
from sklearn import metrics


# In[4]:


#BEGIN HARDCODING OF GUI INFO
n = 2000 / 60
N = 16
Bd = 0.331*254
Pd = 2.815*254
phi = 15.17 * np.pi / 180
SampleFrequency = 20000
FileOfInterest = '2003.10.22.12.06.24'
HomeDirectory = os.getcwd()
os.chdir(HomeDirectory)
directory = os.listdir(HomeDirectory)
TrainingDataFile = "DELETE.csv"
#END OF HARDCODING OF GUI INFO


# In[5]:


UserInput = UserInputs2WorkingForm(n,N,Bd,Pd,phi,SampleFrequency,FileOfInterest,HomeDirectory,directory,TrainingDataFile)
X_train, X_test, Y_train, Y_test = GetTrainingData(UserInput)


# In[6]:


models = GetAllModelsForComparison(X_train,Y_train)
# evaluate each model in turn

results = []
names = []
i = 0
string = []
string1 = []
fig = []
time = []
for ModelName in models:
    before = datetime.now()
    try:
        CTest = models[ModelName].fit(X_train, Y_train)
        #Y_pred = CTest.predict(X_test)
        #results = metrics.classification_report(Y_test,Y_pred)
        #string.append("The accuracy of {} is: {}".format(ModelName,results)) 
        string.append("The accuracy of {} is: {}".format(ModelName,CTest.score(X_test, Y_test)))
        
    except:
        string.append("{} has no available accuracy mark.".format(ModelName))
    end = datetime.now()
    time.append("{} took {} time".format(ModelName,(end-before)))
    try:
        m = CTest.feature_importances_
        m1 = GetTESTDataFrameNames(UserInput)
        Z = [x for _,x in sorted(zip(m,m1))]
        Z1 = sorted(m)
        fig.append(getBarPlot(Z1[-10:],Z[-10:],"Relative Importance",ModelName))
    except:
        string1.append("{} has no feature importance".format(ModelName)) 
     


# In[ ]:


import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("Graphs1.pdf")
i = 0
for figure in fig:
    pdf.savefig( fig[i],dpi=300, bbox_inches = "tight")
    i += 1
pdf.close()

if not(not time):
    with open('Time1.txt', 'w') as writeFile:
        for i in np.arange(len(time)):
            writeFile.write("%(t)s\n" % {"t":time[i]})
writeFile.close()   

if not(not string1):
    with open('NoGraphs1.txt', 'w') as writeFile:
        for i in np.arange(len(string1)):
            writeFile.write("%(t)s\n" % {"t":string1[i]})
writeFile.close()       

if not(not string):
    with open('Accuracy1.txt', 'w') as writeFile:
        for i in np.arange(len(string)):
            writeFile.write("%(t)s\n" % {"t":string[i]})
        
        
writeFile.close()


# In[ ]:


X_train, X_test, Y_train, Y_test = GetTrainingData2(UserInput)


# In[ ]:


models = GetAllModelsForComparison(X_train,Y_train)
# evaluate each model in turn

results = []
names = []
i = 0
string = []
string1 = []
fig = []
time = []
for ModelName in models:
    before = datetime.now()
    try:
        CTest = models[ModelName].fit(X_train, Y_train)
        #Y_pred = CTest.predict(X_test)
        #results = metrics.classification_report(Y_test,Y_pred)
        #string.append("The accuracy of {} is: {}".format(ModelName,results)) 
        string.append("The accuracy of {} is: {}".format(ModelName,CTest.score(X_test, Y_test))) 
        
    except:
        string.append("{} has no available accuracy mark.".format(ModelName))
    end = datetime.now()
    time.append("{} took {} time".format(ModelName,(end-before)))
    try:
        m = CTest.feature_importances_
        m1 = GetTESTDataFrameNames(UserInput)
        Z = [x for _,x in sorted(zip(m,m1))]
        Z1 = sorted(m)
        fig.append(getBarPlot(Z1[-10:],Z[-10:],"Relative Importance",ModelName))
    except:
        string1.append("{} has no feature importance".format(ModelName)) 
     


# In[ ]:


import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("Graphs2.pdf")
i = 0
for figure in fig:
    pdf.savefig( fig[i],dpi=300, bbox_inches = "tight")
    i += 1
pdf.close()

if not(not time):
    with open('Time2.txt', 'w') as writeFile:
        for i in np.arange(len(time)):
            writeFile.write("%(t)s\n" % {"t":time[i]})
writeFile.close()   

if not(not string1):
    with open('NoGraphs2.txt', 'w') as writeFile:
        for i in np.arange(len(string1)):
            writeFile.write("%(t)s\n" % {"t":string1[i]})
writeFile.close()       

if not(not string):
    with open('Accuracy2.txt', 'w') as writeFile:
        for i in np.arange(len(string)):
            writeFile.write("%(t)s\n" % {"t":string[i]})
        
        
writeFile.close()


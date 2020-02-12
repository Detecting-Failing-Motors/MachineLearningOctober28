
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


#Feature Selection
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[4]:


def BearingInfomation(UserInput):
    n = UserInput['n']
    N = UserInput['N']
    Bd = UserInput['Bd']
    Pd = UserInput['Pd']
    phi = UserInput['Phi']
    xx = Bd/Pd*np.cos(phi)
    BPFI = (N/2)*(1 + xx)*n
    BPFO = (N/2)*(1 - xx)*n
    BSF = (Pd/(2*Bd))*(1-(xx)**2)*n
    FTF= (1/2)*(1 - xx)*n
    x = {
        "BPFI": BPFI,
        "BPFO": BPFO,
        "BSF":  BSF,
        "FTF":  FTF
    }
    return x


# In[5]:


def RemoveDCOffset(UserInput):
    UserInput["Signal Data of Interest"] = UserInput["Signal Data of Interest"] - np.mean(UserInput["Signal Data of Interest"])
    return UserInput


# In[6]:


def FourierTransform(UserInput):
    #Fast Fourier Transform
    sig = UserInput['Signal Data of Interest']
    NumberOfSamples = UserInput['Number of Samples']
    Tmax = UserInput['Time of Sampling']
    frq = np.arange(NumberOfSamples)/(Tmax)# two sides frequency range
    frq = frq[range(int(NumberOfSamples/(2)))] # one side frequency range
    Y = abs(np.fft.fft(sig))/NumberOfSamples # fft computing and normalization
    Y = Y[range(int(NumberOfSamples/2))]
    #End fft
    x = {
        "Frequency":frq,
        "Freq. Amp.": Y
        }
    return x


# In[7]:


def get_psd_values(UserInput):
    sig = UserInput['Signal Data of Interest']
    SamplingFrequency = UserInput['Sampling Frequency']
    frq, psd_values = welch(sig, fs=SamplingFrequency)
    x = {
        "Frequency":frq,
        "PSD": psd_values
        }
    return x


# In[8]:


def autocorr(x):
    #Subfunction of get_autocorr_values
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(UserInput):
    sig = UserInput['Signal Data of Interest']
    Tmax = UserInput['Time of Sampling']
    N = UserInput['Number of Samples']
    autocorr_values = autocorr(sig)
    x_values = np.array([Tmax * jj for jj in range(0, N)])
    x = {
        "X Values":x_values,
        "Autocorr Values": autocorr_values
        }
    return x


# In[9]:


def TimeDomainInformation(UserInput):
    sig = UserInput['Signal Data of Interest']
    x = {
        "RMS": np.mean(sig**2),
        "STD": np.std(sig),
        "Mean": np.mean(sig),
        "Max": np.max(sig),
        "Min": np.min(sig),
        "Peak-to-Peak": (np.max(sig) - np.min(sig)),
        "Max ABS": np.max(abs(sig)),
        "Kurtosis": kurtosis(sig),
        "Skew": skew(sig),
    }

    return x


# In[10]:


def GetSortedPeak(X,Y):
    #SubFunction for FrequencyDomainInformation
    max_peak_height = 0.1 * np.nanmax(Y)
    threshold = 0.05 * np.nanmax(Y)
    #Get indices of peak
    peak = detect_peaks(Y,edge = 'rising',mph = max_peak_height, mpd = 2, threshold = threshold )
    
    m = []
    mm = []
    for i in peak:
        m.append(Y[i]) 
        mm.append(X[i])

    mmm = np.argsort(m)
    n = []
    nn = []
    for i in mmm:
        n.append(m[i])
        nn.append(mm[i])

    n  = n[::-1]
    nn = nn[::-1]

    return n, nn

def FrequencyDomainInformation(UserInput):
    x1 = FourierTransform(UserInput)
    x2 = get_psd_values(UserInput)
    x3 = get_autocorr_values(UserInput)
    FTamp,FTfreq = GetSortedPeak(x1['Frequency'],x1['Freq. Amp.'])
    PSDamp,PSDfreq = GetSortedPeak(x2['Frequency'],x2['PSD'])
    Cor,CorTime = GetSortedPeak(x3['X Values'],x3['Autocorr Values'])

    while len(FTamp) <= 5:
        FTamp.append(['-999'])
    while len(FTfreq) <= 5:
        FTfreq.append(['-999'])
    while len(PSDamp) <= 5:
        PSDamp.append(['-999'])
    while len(PSDfreq) <= 5:
        PSDfreq.append(['-999'])
    while len(Cor) <= 5:
        Cor.append(['-999'])
    while len(CorTime) <= 5:
        CorTime.append(['-999'])
    
    x = {
        "FFT Frq @ Peak 1": FTfreq[0],
        "FFT Frq @ Peak 2": FTfreq[1],
        "FFT Frq @ Peak 3": FTfreq[2],
        "FFT Frq @ Peak 4": FTfreq[3],
        "FFT Frq @ Peak 5": FTfreq[4],
        "FFT Amp @ Peak 1": FTamp[0],
        "FFT Amp @ Peak 2": FTamp[1],
        "FFT Amp @ Peak 3": FTamp[2],
        "FFT Amp @ Peak 4": FTamp[3],
        "FFT Amp @ Peak 5": FTamp[4],
        "PSD Frq @ Peak 1": PSDfreq[0],
        "PSD Frq @ Peak 2": PSDfreq[1],
        "PSD Frq @ Peak 3": PSDfreq[2],
        "PSD Frq @ Peak 4": PSDfreq[3],
        "PSD Frq @ Peak 5": PSDfreq[4],
        "PSD Amp @ Peak 1": PSDamp[0],
        "PSD Amp @ Peak 2": PSDamp[1],
        "PSD Amp @ Peak 3": PSDamp[2],
        "PSD Amp @ Peak 4": PSDamp[3],
        "PSD Amp @ Peak 5": PSDamp[4],
        "Autocorrelate Time @ Peak 1": CorTime[0],
        "Autocorrelate Time @ Peak 2": CorTime[1],
        "Autocorrelate Time @ Peak 3": CorTime[2],
        "Autocorrelate Time @ Peak 4": CorTime[3],
        "Autocorrelate Time @ Peak 5": CorTime[4],
        "Autocorrelate @ Peak 1": Cor[0],
        "Autocorrelate @ Peak 2": Cor[1],
        "Autocorrelate @ Peak 3": Cor[2],
        "Autocorrelate @ Peak 4": Cor[3],
        "Autocorrelate @ Peak 5": Cor[4]
    }
    return x


# In[11]:


"""
http://mkalikatzarakis.eu/wp-content/uploads/2018/12/IMS_dset.html
Previous work done on this dataset states that seven different states of health were observed:

Early (initial run-in of the bearings)
Normal
Suspect (the health seems to be deteriorating)
Imminent failure (for bearings 1 and 2, which didn’t actually fail, but were severely worn out)
Inner race failure (bearing 3)
Rolling element failure (bearing 4)
Stage 2 failure (bearing 4)
For the first test (the one we are working on), the following labels have been proposed per file:

Bearing 1
early: 2003.10.22.12.06.24 - 2013.10.23.09.14.13
suspect: 2013.10.23.09.24.13 - 2003.11.08.12.11.44 (bearing 1 was in suspicious health from the beginning, but showed some self-healing effects)
normal: 2003.11.08.12.21.44 - 2003.11.19.21.06.07
suspect: 2003.11.19.21.16.07 - 2003.11.24.20.47.32
imminent failure: 2003.11.24.20.57.32 - 2003.11.25.23.39.56

Bearing 2
early: 2003.10.22.12.06.24 - 2003.11.01.21.41.44
normal: 2003.11.01.21.51.44 - 2003.11.24.01.01.24
suspect: 2003.11.24.01.11.24 - 2003.11.25.10.47.32
imminent failure: 2003.11.25.10.57.32 - 2003.11.25.23.39.56

Bearing 3
early: 2003.10.22.12.06.24 - 2003.11.01.21.41.44
normal: 2003.11.01.21.51.44 - 2003.11.22.09.16.56
suspect: 2003.11.22.09.26.56 - 2003.11.25.10.47.32
Inner race failure: 2003.11.25.10.57.32 - 2003.11.25.23.39.56

Bearing 4
early: 2003.10.22.12.06.24 - 2003.10.29.21.39.46
normal: 2003.10.29.21.49.46 - 2003.11.15.05.08.46
suspect: 2003.11.15.05.18.46 - 2003.11.18.19.12.30
Rolling element failure: 2003.11.19.09.06.09 - 2003.11.22.17.36.56
Stage 2 failure: 2003.11.22.17.46.56 - 2003.11.25.23.39.56
"""

def getAbsoluteTime(file):
    #Subfunction for StateInformation
    year   = int(file[0:4])
    month  = int(file[5:7])
    day    = int(file[8:10])
    hour   = int(file[11:13])
    minute = int(file[14:16])
    second = int(file[17:19])
    x = second + 60*minute + 60*60*hour + 24*60*60*day + 31*24*60*60*(month - 10)
    return x

def StateInformation(UserInput,BearingNum):
    file = UserInput['File of Interest']
    absolutetime = getAbsoluteTime(file)
    #in seconds don't include years taking 10 as the start month
    
    #Bearing 1 transitions
    b1e2s  = getAbsoluteTime("2013.10.23.09.14.13")
    b1s2n  = getAbsoluteTime("2003.11.08.12.11.44")
    b1n2s  = getAbsoluteTime("2003.11.19.21.06.07")
    b1s2i  = getAbsoluteTime("2003.11.24.20.47.32")
    
    #Bearing 2 transitions
    b2e2n  = getAbsoluteTime("2003.11.01.21.41.44")
    b2n2s  = getAbsoluteTime("2003.11.24.01.01.24")
    b2s2i  = getAbsoluteTime("2003.11.25.10.47.32")
    
    #Bearing 3 transitions
    b3e2n  = getAbsoluteTime("2003.11.01.21.41.44")
    b3n2s  = getAbsoluteTime("2003.11.22.09.16.56")
    b3s2irf  = getAbsoluteTime("2003.11.25.10.47.32")
    
    #Bearing 4 transitions
    b4e2n  = getAbsoluteTime("2003.10.29.21.39.46")
    b4n2s  = getAbsoluteTime("2003.11.15.05.08.46")
    b4s2r  = getAbsoluteTime("2003.11.18.19.12.30")
    b4r2f  = getAbsoluteTime("2003.11.22.17.36.56")
    
    m = "ERROR"
    if BearingNum == 1:
        if absolutetime   <= b1e2s:
            m = "Early"
        elif absolutetime <= b1s2n:
            m = "Suspect"
        elif absolutetime <= b1n2s:
            m = "Normal"
        elif absolutetime <= b1s2i:
            m = "Suspect"
        elif absolutetime > b1s2i:
            m = "Imminent Failure"
    elif BearingNum == 2:
        if absolutetime   <= b2e2n:
            m = "Early"
        elif absolutetime <= b2n2s:
            m = "Normal"
        elif absolutetime <= b2s2i:
            m = "Suspect"
        elif absolutetime > b2s2i:
            m = "Imminent Failure" 
    elif BearingNum == 3:
        if absolutetime   <= b3e2n:
            m = "Early"
        elif absolutetime <= b3n2s:
            m = "Normal"
        elif absolutetime <= b3s2irf:
            m = "Suspect"
        elif absolutetime >= b3s2irf:
            m = "Inner Race Failure"   
    elif BearingNum == 4:
        if absolutetime   <= b4e2n:
            m = "Early"
        elif absolutetime <= b4n2s:
            m = "Normal"
        elif absolutetime <= b4s2r:
            m = "Suspect"
        elif absolutetime <= b4r2f:
            m = "Rolling Element Failure"
        elif absolutetime > b4r2f:
            m = "Stage 2 Failure"
    else:
        m = "ERROR"
        
    x = {
        "State": m
    }
    return x


# In[12]:


def MotorInformation(UserInput):
    x = {
        "Motor Type AC(1)-DC(0)": 1,
        "Shaft Speed [Hz]": 2000/60
    }
    return x


# In[13]:


def getCompleteDataFrame(UserInput,BearingNum):
    UserInput1 = UserInput
    UserInput1 = RemoveDCOffset(UserInput1)
    BearingInfo = BearingInfomation(UserInput1)
    TimeDomainInfo = TimeDomainInformation(UserInput1)
    FrequecyDomainInfo = FrequencyDomainInformation(UserInput1)
    StateInfo = StateInformation(UserInput1,BearingNum)
    MotorInfo = MotorInformation(UserInput1)
    Features = {**StateInfo,**MotorInfo,**BearingInfo,**TimeDomainInfo,**FrequecyDomainInfo}
    Features = pd.DataFrame(Features, index=[0])
    return Features 


# In[14]:


def getTESTDataFrame(UserInput):
    UserInput1 = UserInput
    UserInput1 = RemoveDCOffset(UserInput1)
    BearingInfo = BearingInfomation(UserInput1)
    TimeDomainInfo = TimeDomainInformation(UserInput1)
    FrequecyDomainInfo = FrequencyDomainInformation(UserInput1)
    MotorInfo = MotorInformation(UserInput1)
    Features = {**MotorInfo,**BearingInfo,**TimeDomainInfo,**FrequecyDomainInfo}
    Features = pd.DataFrame(Features, index=[0])
    return Features 


# In[15]:


def getPlot(X,Y,xlabel,ylabel,Title):
    #Subfunction of getGraphs
    fig = plt.figure()
    plt.plot(X,Y,c = np.random.rand(3,))
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(Title)
    plt.grid(True)
    return fig

def getGraphs(UserInput):
    t = np.arange(0,UserInput['Time of Sampling'],1/UserInput['Sampling Frequency'])
    figs = []
    x1 = FourierTransform(UserInput)
    x2 = get_psd_values(UserInput)
    x3 = get_autocorr_values(UserInput)
    UserInput1 = RemoveDCOffset(UserInput)
    figs.append(getPlot(t,UserInput['Signal Data of Interest'],"time (s)","Amplitude","Raw Data"))
    figs.append(getPlot(t,UserInput1['Signal Data of Interest'],"time (s)","Amplitude","Raw Data w/ Removed DC Offset"))
    figs.append(getPlot(x1['Frequency'],x1['Freq. Amp.'],'Frequency [Hz]',"time (s)","FFT"))
    figs.append(getPlot(x2['Frequency'],x2['PSD'],'Frequency [Hz]','PSD [V**2 / Hz]',"PSD"))
    figs.append(getPlot(x3['X Values'],x3['Autocorr Values'],'time delay [s]',"Autocorrelation amplitude","Autocorrelation"))

    return figs


# In[16]:


def getBarPlot(X,Y,xlabel,Title):
    #Subfunction of getGraphs
    fig = plt.figure()
    y_pos = np.arange(len(Y))
    plt.barh(y_pos, X, align='center')
    plt.xlabel(xlabel, fontsize=12)
    plt.yticks(y_pos, Y)
    plt.title(Title)
    plt.grid(True)
    return fig


# In[17]:


def GetData(FileOfInterest):
    #Subfunction for UserInputs2WorkingForm
    data = pd.read_table(FileOfInterest,header = None)
    data.columns = ['b1x','b1y','b2x','b2y','b3x','b3y','b4x','b4y']
    return np.transpose(data.values[:,0])

def UserInputs2WorkingForm(n,N,Bd,Pd,phi,SampleFrequency,FileOfInterest,                           HomeDirectory,directory,TrainingDataFile):
    sig = GetData(FileOfInterest)
    NumberOfSamples = len(sig)
    dt = 1/SampleFrequency
    Tmax = dt*NumberOfSamples
    x = {
        'n': n, #Shaft rotational speed [Hz], n
        'N': N, #No. of rolling elements [-], N
        'Bd': Bd, #Diameter of a rolling element [mm], Bd
        'Pd': Pd, #Pitch diameter [mm], Pd
        'Phi': phi, #Contact angle [rad], Phi
        'Sampling Frequency': SampleFrequency,
        'Time of Sampling': Tmax,
        'Number of Samples': NumberOfSamples,
        'File of Interest': FileOfInterest,
        'HomeDirectory': HomeDirectory,
        'Working Directory': directory,
        'TrainingFileName': TrainingDataFile,
        'Signal Data of Interest': sig    
    }
    return x


# In[18]:


def truncate(f, n):
    '''https://stackoverflow.com/questions/783897/truncating-floats-in-python/51172324#51172324'''
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


# In[19]:


def GetTrainingData(UserInput):
    for file in UserInput['Working Directory']:
        if file == UserInput['TrainingFileName']:
            dataset = pd.read_csv(file,header = 0,index_col = 0)

    X = dataset.values[:,1:(dataset.shape[1]-1)]
    Y = dataset.values[:,0]
    validation_size = 0.20
    seed = 6
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) 
    
    return X_train, X_test, Y_train, Y_test


# In[20]:


def GetTrainingData2(UserInput):
    for file in UserInput['Working Directory']:
        if file == UserInput['TrainingFileName']:
            dataset = pd.read_csv(file,header = 0,index_col = 0)

    X = dataset.values[:,1:(dataset.shape[1]-1)]
    Y = dataset.values[:,0]
    validation_size = 0.20
    seed = 21
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) 
    
    return X_train, X_test, Y_train, Y_test


# In[21]:


def GetTESTDataFrameNames(UserInput):
    for file in UserInput['Working Directory']:
        if file == UserInput['TrainingFileName']:
            dataset = pd.read_csv(file,header = 0,index_col = 0)
    names = []
    for x in dataset.columns:
        names.append(x)
    return names


# In[22]:


def TrainModel(X_train,Y_train):
    classifier = RandomForestClassifier(n_estimators=1000)
    classifier = classifier.fit(X_train, Y_train)
    return classifier


# In[23]:


def PredictModel(classifier,X_test):
    Y_test_pred = classifier.predict(X_test)
    return Y_test_pred


# In[24]:


def GetAllModelsForComparison(X_train,Y_train):
    models = {
        'ARDRegression': ARDRegression(),
        'BayesianRidge': BayesianRidge(),
        'ElasticNet': ElasticNet(),
        'ElasticNetCV': ElasticNetCV(),
        'Hinge': Hinge(),
        #'Huber': Huber(),
        'HuberRegressor': HuberRegressor(),
        'Lars': Lars(),
        'LarsCV': LarsCV(),
        'Lasso': Lasso(),
        'LassoCV': LassoCV(),
        'LassoLars': LassoLars(),
        'LassoLarsCV': LassoLarsCV(),
        'LinearRegression': LinearRegression(),
        'Log': Log(),
        'LogisticRegression': LogisticRegression(),
        'LogisticRegressionCV': LogisticRegressionCV(),
        'ModifiedHuber': ModifiedHuber(),
        'MultiTaskElasticNet': MultiTaskElasticNet(),
        'MultiTaskElasticNetCV': MultiTaskElasticNetCV(),
        'MultiTaskLasso': MultiTaskLasso(),
        'MultiTaskLassoCV': MultiTaskLassoCV(),
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
        'OrthogonalMatchingPursuitCV': OrthogonalMatchingPursuitCV(),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
        'Perceptron': Perceptron(),
        'RANSACRegressor': RANSACRegressor(),
        #'RandomizedLasso': RandomizedLasso(),
        #'RandomizedLogisticRegression': RandomizedLogisticRegression(),
        'Ridge': Ridge(),
        'RidgeCV': RidgeCV(),
        'RidgeClassifier': RidgeClassifier(),
        'SGDClassifier': SGDClassifier(),
        'SGDRegressor': SGDRegressor(),
        'SquaredLoss': SquaredLoss(),
        'TheilSenRegressor': TheilSenRegressor(),
        'BaseEstimator': BaseEstimator(),
        'ClassifierMixin': ClassifierMixin(),
        'LinearClassifierMixin': LinearClassifierMixin(),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'StandardScaler': StandardScaler(),
        'TransformerMixin': TransformerMixin(),
        'BaseEstimator': BaseEstimator(),
        'KernelRidge': KernelRidge(),
        'RegressorMixin': RegressorMixin(),
        'LinearSVC': LinearSVC(),
        'LinearSVR': LinearSVR(),
        'NuSVC': NuSVC(),
        'NuSVR': NuSVR(),
        'OneClassSVM': OneClassSVM(),
        'SVC': SVC(),
        'SVR': SVR(),
        'SGDClassifier': SGDClassifier(),
        'SGDRegressor': SGDRegressor(),
        #'BallTree': BallTree(),
        #'DistanceMetric': DistanceMetric(),
        #'KDTree': KDTree(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'KernelDensity': KernelDensity(),
        #'LSHForest': LSHForest(),
        'LocalOutlierFactor': LocalOutlierFactor(),
        'NearestCentroid': NearestCentroid(),
        'NearestNeighbors': NearestNeighbors(),
        'RadiusNeighborsClassifier': RadiusNeighborsClassifier(),
        'RadiusNeighborsRegressor': RadiusNeighborsRegressor(),
        #'GaussianProcess': GaussianProcess(),
        'GaussianProcessRegressor': GaussianProcessRegressor(),
        'GaussianProcessClassifier': GaussianProcessClassifier(),
        'CCA': CCA(),
        'PLSCanonical': PLSCanonical(),
        'PLSRegression': PLSRegression(),
        'PLSSVD': PLSSVD(),
        #'ABCMeta': ABCMeta(),
        #'BaseDiscreteNB': BaseDiscreteNB(),
        'BaseEstimator': BaseEstimator(),
        #'BaseNB': BaseNB(),
        'BernoulliNB': BernoulliNB(),
        'ClassifierMixin': ClassifierMixin(),
        'GaussianNB': GaussianNB(),
        'LabelBinarizer': LabelBinarizer(),
        'MultinomialNB': MultinomialNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'ExtraTreeClassifier': ExtraTreeClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'BaggingClassifier': BaggingClassifier(),
        'BaggingRegressor': BaggingRegressor(),
        #'BaseEnsemble': BaseEnsemble(),
        'ExtraTreesClassifier': ExtraTreesClassifier(),
        'ExtraTreesRegressor': ExtraTreesRegressor(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'IsolationForest': IsolationForest(),
        'RandomForestClassifier': RandomForestClassifier(),
        'RandomForestRegressor': RandomForestRegressor(),
        'RandomTreesEmbedding': RandomTreesEmbedding(),
        #'VotingClassifier': VotingClassifier(),
        'BaseEstimator': BaseEstimator(),
        'ClassifierMixin': ClassifierMixin(),
        'LabelBinarizer': LabelBinarizer(),
        'MetaEstimatorMixin': MetaEstimatorMixin(),
        #'OneVsOneClassifier': OneVsOneClassifier(),
        #'OneVsRestClassifier': OneVsRestClassifier(),
        #'OutputCodeClassifier': OutputCodeClassifier(),
        'Parallel': Parallel(),
        #'ABCMeta': ABCMeta(),
        'BaseEstimator': BaseEstimator(),
        #'ClassifierChain': ClassifierChain(),
        'ClassifierMixin': ClassifierMixin(),
        'MetaEstimatorMixin': MetaEstimatorMixin(),
        #'MultiOutputClassifier': MultiOutputClassifier(),
        #'MultiOutputEstimator': MultiOutputEstimator(),
        #'MultiOutputRegressor': MultiOutputRegressor(),
        'Parallel': Parallel(),
        'RegressorMixin': RegressorMixin(),
        'LabelPropagation': LabelPropagation(),
        'LabelSpreading': LabelSpreading(),
        'BaseEstimator': BaseEstimator(),
        'IsotonicRegression': IsotonicRegression(),
        'RegressorMixin': RegressorMixin(),
        'TransformerMixin': TransformerMixin(),
        'BernoulliRBM': BernoulliRBM(),
        'MLPClassifier': MLPClassifier(),
        'MLPRegressor': MLPRegressor()
        }
    return models


# In[25]:


def GetUserInputNames(UserInput):
    names = []
    for x in UserInput:
        names.append(x)
    return names


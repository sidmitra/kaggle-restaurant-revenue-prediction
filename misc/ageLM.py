from sklearn.linear_model import LinearRegression as LM
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from random import shuffle

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#calculate the age of each sample. fyi this only worked with pandas 0.14, not 0.13
train['Age']=2015-pd.DatetimeIndex(train['Open Date']).year
test['Age']=2015-pd.DatetimeIndex(test['Open Date']).year

#Extract the age and log transform it
X=np.log(train[['Age']].values.reshape((train.shape[0],1)))
Xt=np.log(test[['Age']].values.reshape((test.shape[0],1)))
y=train['revenue'].values

#randomize the order for cross validation
combined=zip(y,X)
shuffle(combined)
y[:], X[:] = zip(*combined)


#Model Setup
clf=LM()

scores=[]

ss=KFold(len(y), n_folds=3,shuffle=True)
for trainCV, testCV in ss:
    X_train, X_test, y_train, y_test= X[trainCV], X[testCV], y[trainCV], y[testCV]
    clf.fit(X_train, np.log(y_train))
    y_pred=np.exp(clf.predict(X_test))

    scores.append(mean_squared_error(y_test,y_pred))

#Average RMSE from cross validation
scores=np.array(scores)
print "CV Score:",np.mean(scores**0.5)

#Fit model again on the full training set
clf.fit(X,np.log(y))
#Predict test.csv & reverse the log transform
yp=np.exp(clf.predict(Xt))

#Write submission file
sub=pd.read_csv('sampleSubmission.csv')
sub['Prediction']=yp
sub.to_csv('sub.csv',index=False)


"""
ss = cross_validation.LeaveOneOut(137)
for trainCV, testCV in ss:
X_train, X_test, y_train, y_test= X[trainCV], X[testCV], y[trainCV], y[testCV]
clf.fit(X_train, np.log(y_train))
y_pred=np.exp(clf.predict(X_test))scores.append(mean_squared_error(y_test,y_pred))#Average RMSE from cross validation
scores=np.array(scores)
print "CV Score:",np.mean(scores**0.5)
"""
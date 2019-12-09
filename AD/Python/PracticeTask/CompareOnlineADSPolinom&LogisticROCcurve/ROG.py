#TODO: CREATE polinomial regression

import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt

#from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

df=pd.read_excel("ONLINEADS.xlsx",sheet_name = 0)
x=df[['Age', 'Gender', 'Interest','Device','AdsTool','VisitNumber']]
y=df['Registration']
y=y.astype('int')

n_estimator = 10
#x, y = make_classification(n_samples=80000)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train, y_train_lr = train_test_split(
    X_train, y_train, test_size=0.5)


poly =PolynomialFeatures(degree=2) 
X_poly =poly.fit_transform(x) 
poly.fit(X_poly, y) 
lin2 =LinearRegression() 
lin2.fit(X_poly, y)


# CREATE ALL TYPES ALGORYTHMS
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder(categories='auto')
rf_lm = LogisticRegression(solver='lbfgs', max_iter=1000)

#FIT TRAIN DATA IN ALL TYPES
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

#DECISION TREE
decTree = DecisionTreeClassifier(max_depth=4, random_state=0) #create dec tree
decTree.fit(X_train, y_train) # fit data to tree


#CURVE FOR LOGISTIC REG
y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1] #create predictions
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm) #create curve


# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

# The decision tree model by itself
y_pred_dt = decTree.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)

#CURVE FOR POLINOMIAL
y_pred_pr = lin2.predict(poly.fit_transform(X_test))
fpr_pr, tpr_pr, _ = roc_curve(y_test, y_pred_pr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_pr, tpr_pr, label = 'PR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_dt, tpr_dt, label='DT')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.xlim(0, 0.5)
plt.ylim(0.7, 1)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(x, lin2.predict(poly.fit_transform(x)), label = 'PR')
plt.plot(fpr_pr, tpr_pr, label = 'PR')

plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_dt, tpr_dt, label='DT')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
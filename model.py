import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_excel('Historical Alarm Cases.xlsx')
df = df.drop(columns=['Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Case No.'])

# there no normal distribution then we use IQR maethod
# if it is Normal Distribution then we used Z-score method
per25 = df['detected by(% of sensors)'].quantile(0.25)
per75 = df['detected by(% of sensors)'].quantile(0.75)
iqr = per75 - per25
lowerlimit = per25 - 1.5 * iqr
upperlimit = per75 + 1.5 * iqr

df['detected by(% of sensors)'] = np.where(df['detected by(% of sensors)'] < lowerlimit, lowerlimit,
                                                np.where(df['detected by(% of sensors)'] > upperlimit,upperlimit,df['detected by(% of sensors)']))

X = df.drop(columns=['Spuriosity Index(0/1)'])
y = df['Spuriosity Index(0/1)']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=50)

trf1 = ColumnTransformer([('scaler',StandardScaler(),slice(0,6))],remainder='passthrough')
trf2 = LogisticRegression()

pipe = Pipeline([('scale',trf1),('Classifier',trf2)])
pipe.fit(X_train,y_train)
# print(pipe.score(X_test,y_test))

parameters = {
    'Classifier__C':[0.001, 0.01, 0.1, 1, 10, 100],
    'Classifier__penalty':['l1','l2','elasticnet'],
    'Classifier__solver': ['liblinear','newton-cholesky']
}

gsc = GridSearchCV(pipe,param_grid=parameters,cv=5,scoring='accuracy')

gsc.fit(X_train,y_train)
y_pred = gsc.predict(X_test)

# print(accuracy_score(y_test,y_pred))

pickle.dump(gsc,open('model.pkl','wb'))
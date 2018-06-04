#importing libraries
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#loading data
path="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data=pd.read_csv(path,header=None,names=['sepal_length','sepal_width','petal_length','petal_width','class'])
train,test = train_test_split(data,test_size=0.2,random_state=7)
#analysing data
print "train shape:   "
print train.shape
print "test shape:    "
print test.shape
print "Unique classes with count : "
print pd.value_counts(train['class'])
print "data description : "
print train.describe()

#checking missing values
print train.info()
#No null values

#Exploratory data analysis
print "Univariate analysis"
print "Target Class"
plt.hist(train['class'])
plt.show()
print "Sepal_length"
sns.distplot(train['sepal_length'],kde=True,bins=40)
plt.show()
print "Sepal_width"
sns.distplot(train['sepal_width'],kde=True,bins=40)
plt.show()
print "Petal_length"
sns.distplot(train['petal_length'],kde=True,bins=40)
plt.show()
print "Petal_width"
sns.distplot(train['petal_width'],kde=True,bins=40)
plt.show()
print "bivariate analysis"
sns.boxplot(x='class',y='sepal_length',data=train,palette='OrRd')
plt.show()
sns.boxplot(x='class',y='sepal_width',data=train,palette='OrRd')
plt.show()
sns.boxplot(x='class',y='petal_length',data=train,palette='OrRd')
plt.show()
sns.boxplot(x='class',y='petal_width',data=train,palette='OrRd')
plt.show()
sns.heatmap(data.corr(),cmap="OrRd", linecolor='white', linewidths=1)
plt.show()
sns.pairplot(train, hue='class',palette='OrRd')
plt.show()

#Modelling
train['class'][train['class']=='Iris-versicolor']=0
train['class'][train['class']=='Iris-setosa']=1
train['class'][train['class']=='Iris-virginica']=2
test['class'][test['class']=='Iris-versicolor']=0
test['class'][test['class']=='Iris-setosa']=1
test['class'][test['class']=='Iris-virginica']=2
X = train.iloc[:,:-1]
y = train.iloc[:,-1]
y = pd.to_numeric(y)
y_test=pd.to_numeric(test.iloc[:,-1])



# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
scoring = 'accuracy'


results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=7)
	cv_results = model_selection.cross_val_score(model, X,y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

#lets make prediction on test data

# Make predictions on validation dataset
print "SVC Performance"
SVM = SVC()
SVM.fit(X,y)
predictions = SVM.predict(test.iloc[:,:-1])
print (accuracy_score(y_test, predictions))
print (confusion_matrix(y_test, predictions))
print (classification_report(y_test, predictions))

print "KNN Performance"
KNN = KNeighborsClassifier()
KNN.fit(X,y)
predictions = KNN.predict(test.iloc[:,:-1])
print (accuracy_score(y_test, predictions))
print (confusion_matrix(y_test, predictions))
print (classification_report(y_test, predictions))

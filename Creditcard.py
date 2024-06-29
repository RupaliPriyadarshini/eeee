# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')


# Loading and exploring the data
df = pd.read_csv('creditcard.csv')
df.head()

df.shape
df.columns

df.info()
df.describe()

df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df.shape

df['Class'].unique()
df['Class'].value_counts()

fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]
f = len(fraud)
n = len(normal)
fPercent = f/(f+n)
nPercent = n/(f+n)
print('Fraud transactions percentage = ', round(fPercent * 100, 2))
print('Normal transactions percentage = ', round(nPercent * 100, 2))

# Analyzing the data
plt.figure(figsize=(6,8))
sns.countplot(data=df, x='Class', hue= 'Class', palette=['purple','red'])
plt.title('No. of Normal & Fraud Transactions')

plt.figure(figsize=(6,6))
sns.FacetGrid(df, hue='Class', height=6,palette=['purple','red']).map(plt.scatter, "Time", "Amount").add_legend()
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(data=df.corr(),cmap='coolwarm')
plt.show()

# Building model
X = df.drop('Class',axis=1)
Y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

def model_train_test(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    print('Accuracy = {}'.format(accuracy_score(y_test,prediction)))
    print(classification_report(y_test,prediction))
    matrix = confusion_matrix(y_test,prediction)
    dis = ConfusionMatrixDisplay(matrix)
    dis.plot()
    plt.show()

model_ = RandomForestClassifier()
model_train_test(model_, X_train, y_train, X_test, y_test)

Decision_tree = DecisionTreeClassifier()
model_train_test(Decision_tree,X_train,y_train,X_test,y_test)
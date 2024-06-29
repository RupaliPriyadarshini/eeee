# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Loading and exploring data
df = pd.read_csv('IRIS.csv')

df.shape
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df.shape
df.info()
df.describe()

# Analysing data
df.head()

spcounts = df['species'].value_counts()

plt.figure(figsize=(10,10))
spcounts.plot(kind='pie', autopct='%1.2f%%', startangle=100)
plt.title('Species')
plt.legend(loc='upper left', labels=spcounts.index)
plt.show()

plt.scatter(df['sepal_length'], df['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title("Sepal's Length and Width")
plt.show()

sns.lmplot(x="sepal_length",y="sepal_width",hue="species",palette="viridis",data=df)
plt.title("Sepal's Length VS Width")
plt.show()

plt.scatter(df['petal_length'], df['petal_width'])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title("Petal's Length and Width")
plt.show()

sns.lmplot(x="petal_length",y="petal_width",hue="species",palette="viridis",data=df)
plt.title("Petal's Length VS Width")
plt.show()

# Building model
label = LabelEncoder()
df['species'] = label.fit_transform(df['species'])

df.head()

x = df.drop('species', axis=1)
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=45)

df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='mako', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

Logistic = LogisticRegression(max_iter=10)
Logistic.fit(x_train,y_train)

Logistic.score(x_train, y_train)

y_predict = Logistic.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

print(accuracy)

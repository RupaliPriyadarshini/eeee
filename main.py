# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Loading and exploring the data
df = pd.read_csv('Titanic-Dataset.csv')

df.head()

df.duplicated().sum()

df.isnull().sum()

df.groupby('Sex')['Age'].mean().reset_index()
df['Age'].fillna(df['Age'].mean(), inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df.dropna(inplace=True)
df.info()

# Analysing the data
df.head()
gender = df['Sex'].value_counts()

plt.figure(figsize=(8,6))
plt.pie(gender, labels=['Male', 'Female'],autopct='%.1f%%', colors=['blue','pink'])
plt.legend()
plt.title('Male and Female')
plt.show()

sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

sur_sex = df[['Survived','Sex']].value_counts().reset_index()

plt.figure(figsize=(8,6))
sns.barplot(data=sur_sex, x=sur_sex['Survived'], y=sur_sex['count'], hue=sur_sex['Sex'])
plt.title('Survived Frequency')
plt.xlabel('Survived')
plt.ylabel('Frequency')
plt.show()

Emb_sex = df[['Embarked', 'Sex']].value_counts().reset_index()

plt.figure(figsize=(8,6))
sns.barplot(data=Emb_sex, x=Emb_sex['Embarked'], y=Emb_sex['count'], hue=Emb_sex['Sex'])
plt.title('Embarked & Sex Frequency')
plt.xlabel('Embarked')
plt.ylabel('Frequency')
plt.show()

sur_emb = df[['Survived', 'Embarked']].value_counts().reset_index()

plt.figure(figsize=(8,6))
sns.barplot(data=sur_emb, x=sur_emb['Survived'], y=sur_emb['count'], hue=sur_emb['Embarked'])
plt.title('Survived & Embarked Frequency')
plt.xlabel('Survived')
plt.ylabel('Frequency')
plt.show()

sur_class = df[['Survived', 'Pclass']].value_counts().reset_index()

plt.figure(figsize=(8,6))
sns.barplot(data=sur_class, x=sur_class['Survived'], y=sur_class['count'], hue=sur_class['Pclass'])
plt.title('Survived & Pclass Frequency')
plt.xlabel('Survived')
plt.ylabel('Frequency')
plt.show()

siblings = df['SibSp'].value_counts().reset_index()

plt.figure(figsize=(8,6))
sns.barplot(x=siblings['SibSp'], y=siblings['count'])
plt.title('Siblings or Spouses Frequency')
plt.show()

sur_siblings = df[['Survived', 'SibSp']].value_counts().reset_index()

plt.figure(figsize=(8,6))
sns.barplot(data=sur_siblings, x=sur_siblings['Survived'], y=sur_siblings['count'], hue=sur_siblings['SibSp'])
plt.title('Survived & SibSp Frequency')
plt.legend(loc='upper right')
plt.xlabel('Survived')
plt.ylabel('Frequency')
plt.show()

# Building model

test = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

label_encoder = LabelEncoder()
test['Sex'] = label_encoder.fit_transform(test['Sex'])
test['Embarked'] = label_encoder.fit_transform(test['Embarked'])

x = test.drop('Survived', axis=1)
y = test['Survived']
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)

model_logistic = LogisticRegression()
model_logistic.fit(X_train,Y_train)

model_logistic.score(X_train,Y_train)

model_logistic.score(x_test,y_test)

model_random = RandomForestClassifier()
model_random.fit(X_train,Y_train)

model_random.score(X_train,Y_train)

model_random.score(x_test,y_test)

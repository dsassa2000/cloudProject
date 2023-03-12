import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

#Evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


df = pd.read_csv("C:/Users/HP/Downloads/Heart_disease_dataset.csv")
df.shape
print(df);
#check missing values of all features
miss_values = df.isna().sum()
print(miss_values);

#how many class of one feature or target
print(df["HeartDiseaseorAttack"].value_counts())

#bar chart   
df["HeartDiseaseorAttack"].value_counts().plot(kind='bar', color=["salmon","lightblue"])

#Compare target and sex column
print(pd.crosstab(df.HeartDiseaseorAttack, df.Sex))

#Create plot of crosstab
pd.crosstab(df.HeartDiseaseorAttack, df.Sex).plot(kind="bar",figsize=(10,6),color=["salmon","lightblue"])
plt.title("Heart Disease Frerquency for Sex")
plt.xlabel("0 = No Disease, 1=Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"]);
plt.xticks(rotation=0);


print(pd.crosstab(df.HeartDiseaseorAttack, df.Smoker))

print(pd.crosstab(df.HeartDiseaseorAttack, df.HighChol))

print(pd.crosstab(df.HeartDiseaseorAttack, df.HighBP))

print(pd.crosstab(df.HeartDiseaseorAttack, df.HvyAlcoholConsump)) # a eliminer

print(pd.crosstab(df.HeartDiseaseorAttack, df.Age))

print(pd.crosstab(df.HeartDiseaseorAttack, df.Stroke)) # a eliminer 

print(pd.crosstab(df.HeartDiseaseorAttack, df.DiffWalk))

print(pd.crosstab(df.HeartDiseaseorAttack, df.GenHlth))



#Make a correlation matrix
print(df.corr())

#Visualise correlation
corr_matrix = df.corr()
fig,ax = plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_matrix,annot=True,linewidths=0.5,fmt=".2f",cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top-0.5)

plt.show()

logistic_features = ['Stroke','HighBP','Age','DiffWalk','GenHlth']
x_logistic = df[logistic_features].values
y_logistic = df['HeartDiseaseorAttack'].values
X_train,X_test,y_train,y_test = train_test_split(x_logistic,y_logistic,test_size=0.2)

model = LogisticRegression()

#Create function to fit and score models
def fit_and_score(model, X_train,X_test,y_train,y_test):
    #dictionary to keep model scores
    model_scores = {}
    #fit model
    model.fit(X_train, y_train)
    #evaluate model and append score
    model_scores=model.score(X_test, y_test)
    return model_scores


model_scores = fit_and_score(model=model, X_train=x_logistic,X_test=X_test, y_train=y_logistic,y_test=y_test)
print("model_scores = ",model_scores)

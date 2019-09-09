#import dependencies
import pandas as pd

#Loading the  titanic dataset
url="train.csv"
df = pd.read_csv(url)
print(df.head())
include=["Age","Sex","Embarked","Survived"]
df_=df[include]

#Filling null values in data
categories=[]
for col,col_type in df_.dtypes.iteritems():
    if col_type=="O":
        categories.append(col)
    else:
        df_[col].fillna(0,inplace=True)
    
#Label Encoding (Coverting Char columns to numerical using one hot encoder)
df_ohe =pd.get_dummies(df_,columns=categories,dummy_na=True)

#Building Model
from sklearn.linear_model import LogisticRegression
dependent_variable="Survived"
x=df_ohe[df_ohe.columns.difference([dependent_variable])]
y=df_ohe[dependent_variable]
lr=LogisticRegression()
lr.fit(x,y)


#Serializing/pickling the model using joblib
from sklearn.externals import joblib
joblib.dump(lr,"model.pkl")
print("Model Dumped Success!!")

#Loading the model we save
lr=joblib.load("model.pkl")

#Saving the data columns from training
model_columns=list(x.columns)
joblib.dump(model_columns,"model_columns.pkl")
print("Models columns dumped!")














import numpy as numpy 
import pandas as pd 
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay,precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer 

df = pd.read_csv('C:/Users/ASUS/Downloads/news/news.csv') 
df.shape 
print(df.head()) 

df.describe() 

labels = df.label 
labels.head() 

x_train, x_test, y_train, y_test = train_test_split(df['text'],labels,test_size=0.2,random_state=7)

vec= TfidfVectorizer(stop_words = 'english', max_df =0.7)
vec_train = vec.fit_transform(x_train)
vec_test = vec.transform(x_test) 

pac = PassiveAggressiveClassifier(max_iter =50)
pac.fit(vec_train,y_train)
y_pred = pac.predict(vec_test) 
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100, 2)}%')

confusion_matrix(y_test,y_pred, labels=['Fake', 'Real']) 


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

credi_hile=pd.read_csv("creditcard.csv")
c=credi_hile.copy()
c=c.dropna()
print(c.head()) #verimizin ilk beş satırını yazdırdık
print(c.info()) #dataframe'ların özetini yazdırdık
print(c.shape) #boyutunu yazdırdık
print(c["Class"].value_counts()) #0'lar ve 1'lerin sayısını yazdırdık
print(c.describe().T)
y=c["Class"]
print(y.head()) #y'nin yani class kolonunun ilk 5 satırını yazdırdık


X=c.drop(["Class"],axis=1)
print(X.head()) #class kolonu dışındaki kolonların ilk 5 satırını yazdırdık

fig=plt.figure(figsize=(10,10))
sns.heatmap(c.corr())
plt.show()

#scatter plot
cols=['Amount','Time']
sns.pairplot(c[cols])
plt.show()


#histogram grafiklerini çizdirdik
c.hist(figsize=(10,10),color='blue')
plt.show()

#test ve train işlemleri
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)






#nb algoritması
nb=GaussianNB()
nb_model=nb.fit(X_train,y_train)
print(nb_model.predict(X_test)[0:10])#test vektörü üzerinde sınıflandırma
print(nb_model.predict_proba(X_test)[0:10])#test vektörü için olasılık tahminleri döndürür
y_pred=nb_model.predict(X_test)
print(cross_val_score(nb_model,X_test,y_test,cv=10))#çapraz doğrulama ile sonuçları yazdırdık
print(cross_val_score(nb_model,X_test,y_test,cv=10).mean())
print(confusion_matrix(y_test,y_pred))#confusion matrixi yazdırdık
print("nb ile accuracy score:")
print(accuracy_score(y_test,y_pred))#accuracy score 0.993 çıktı

#logistic regression
logreg= LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(" Logistic Regression ile accuracy score:")
print(accuracy_score(y_test, y_pred))#accuracy score 0.998

#Decision Tree
DT = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
print("Decision Tree model ile accuracy score:")
print(accuracy_score(y_test, y_pred))#accuracy score 0.999 çıktı


#Güvenilir ve dolandırıcı labelleri olacak şekilde confusion matrixini çizdirdik
labels= ['Güvenilir', 'Dolandirici']
conf_matrix=confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title(" Confusion Matrix")
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()

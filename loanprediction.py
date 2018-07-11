#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix


#Data Importing
loan=pd.read_csv('loanprediction_train.csv')
print(loan.head(5))

print(loan.info())
print(loan.describe())

'''Exploratory
   Data
   Analysis
'''
#sns.pairplot(loan)
#plt.show()

loan['ApplicantIncome'].hist(bins=30)
plt.xlabel('ApplicantIncome')
plt.ylabel('Frequency')
plt.show()

loan['CoapplicantIncome'].hist(bins=30)
plt.xlabel('CoapplicantIncome')
plt.ylabel('Frequency')
plt.show()

loan['LoanAmount'].hist(bins=30)
plt.xlabel('LoanAMount')
plt.ylabel('Frequency')
plt.show()

loan['Loan_Amount_Term'].hist(bins=30)
plt.xlabel('LoanAmountTerm')
plt.ylabel('Frequency')
plt.show()

print('Unique values of loan amount term :')
print(loan['Loan_Amount_Term'].unique())

sns.boxplot(x="Education",y="ApplicantIncome",data=loan)
plt.show()

sns.boxplot(x="Education",y="CoapplicantIncome",data=loan)
plt.show()

sns.boxplot(x="Education",y="LoanAmount",data=loan)
plt.show()

#No. of outliers for graduates are more for above three variables while the median value of graduate and non graduate is comparable

print('Frequency table for Credit History')
print(loan['Credit_History'].value_counts())

d={'Y':1,'N':0}
loan['Loan_Status']=loan['Loan_Status'].map(d)

print('Frequency table for Loan Status')
print(loan['Loan_Status'].value_counts())

#Probability of getting loan wrt Credit History
temp=loan.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.mean())
print('Probability of getting loan wrt Credit History:')
print(temp)

#Probability of getting loan wrt Dependents
temp1=loan.pivot_table(values='Loan_Status',index=['Dependents'],aggfunc=lambda x: x.mean())
print('Probability of getting loan wrt Dependents:')
print(temp1)

#Bar plot for credit history vs loan status
temp.plot(kind = 'bar')
plt.show()

'''
   Data 
   Preprocessing
'''

'''
   Missing 
   Value
   Treatment
'''

print('No. of missing values for each variables...')
print(loan.apply(lambda x: sum(x.isnull()),axis=0))

print('Imputing missing value for LoanAmount variable')
loan['LoanAmount'].fillna(loan['LoanAmount'].mean(), inplace=True)
print(loan['LoanAmount'].describe())

#Checking for missing values in Gender variable
print('Frequency table of gender')
print(loan['Gender'].value_counts())

loan['Gender'].fillna('Male',inplace=True)
print(loan['Gender'].value_counts())

#Checking for missing values in Married column 
print(loan['Married'].value_counts())
loan['Married'].fillna('Yes',inplace=True)

#Checking missing values in Dependents
print(loan['Dependents'].value_counts())
loan['Dependents'].fillna(0,inplace=True)
print(loan['Dependents'].describe())

#Checking missing value in Self_Employed variable
print(loan['Self_Employed'].value_counts())
loan['Self_Employed'].fillna('No',inplace=True)
print(loan['Self_Employed'].describe())

#Checking missing value in Loan_Amount_Term
print(loan['Loan_Amount_Term'].value_counts())
loan['Loan_Amount_Term'].fillna(360,inplace=True)
print(loan['Loan_Amount_Term'].describe())

#Missing value treatment for Credit History
print(loan['Credit_History'].value_counts())
loan['Credit_History'].fillna(1,inplace=True)

print(loan.apply(lambda x: sum(x.isnull()),axis=0))


'''
   Outliers
   Treatment
'''

#Earlier during initial data analysis we found that ApplicantIncome,CoapplicantIncome and Loan_Amount had lots of outliers

loan['LoanAmount'].hist(bins=30)
plt.title('LoanAmount distribution')
plt.show()

loan['LoanAmount_log']=np.log(loan['LoanAmount'])
loan['LoanAmount_sq']=np.sqrt(loan['LoanAmount'])

loan['LoanAmount_log'].hist(bins=30)
plt.title('LoanAmount_log distribution')
plt.show()

loan['LoanAmount_sq'].hist(bins=30)
plt.title('LoanAmount_sq Distribution')
plt.show()

loan['TotalIncome']=loan['ApplicantIncome']+loan['CoapplicantIncome']

loan['TotalIncome'].hist(bins=30)
plt.title('Total Income Distribution')
plt.show()

loan['TotalIncome_log']=np.log(loan['TotalIncome'])
loan['TotalIncome_log'].hist(bins=30)
plt.title('TotalIncome_log Distribution')
plt.show()

loan['TotalIncome_sq']=np.sqrt(loan['TotalIncome'])
loan['TotalIncome_sq'].hist(bins=30)
plt.title('TotalIncome_sqrt Distribution')
plt.show()

#One Hot Encoding
'''
catFeat=['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le=LabelEncoder()
oe=OneHotEncoder()
for i in catFeat:
    loan[i]=le.fit_transform(loan[i])
    

print(loan.head(5))
'''

#catFeat=['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
#pd.get_dummies(loan[catFeat])

#print(loan.head(5))

loan_dummy=pd.get_dummies(data=loan, columns=['Gender','Married','Dependents','Education','Self_Employed','Property_Area'])

#print(loan_dummy.head())

#modelfeatures=['Loan_Amount_Term','Credit_History','LoanAmount_log','TotalIncome_log','Dependents_1','Dependents_2','Dependents_3+','Education_Graduate','Education_NotGraduate','Self_Employed_No','Self_Employed_Yes','Property_Area_Rural','Property_Area_Semiurban','Property_Area_Urban']



print(loan_dummy.info())

feat1=loan_dummy[['LoanAmount','Loan_Amount_Term','Credit_History','TotalIncome','Gender_Female','Gender_Male','Married_No','Married_Yes','Dependents_0','Dependents_1','Dependents_2','Dependents_3+','Education_Graduate','Education_Not Graduate','Self_Employed_No','Self_Employed_Yes','Property_Area_Rural','Property_Area_Semiurban','Property_Area_Urban']]

feat2=loan_dummy[['LoanAmount_log','Loan_Amount_Term','Credit_History','TotalIncome_log','Gender_Female','Gender_Male','Married_No','Married_Yes','Dependents_0','Dependents_1','Dependents_2','Dependents_3+','Education_Graduate','Education_Not Graduate','Self_Employed_No','Self_Employed_Yes','Property_Area_Rural','Property_Area_Semiurban','Property_Area_Urban']]

feat3=loan_dummy[['LoanAmount_sq','Loan_Amount_Term','Credit_History','TotalIncome_sq','Gender_Female','Gender_Male','Married_No','Married_Yes','Dependents_0','Dependents_1','Dependents_2','Dependents_3+','Education_Graduate','Education_Not Graduate','Self_Employed_No','Self_Employed_Yes','Property_Area_Rural','Property_Area_Semiurban','Property_Area_Urban']]

outcome=loan_dummy['Loan_Status']

#Logistic Regression Modelling

model1=LogisticRegression()
model1.fit(feat1,outcome)

#Test data manipulation
loan_test=pd.read_csv('loanprediction_test.csv')
print(loan_test.head(5))

loan_test['LoanAmount'].fillna(loan_test['LoanAmount'].mean(), inplace=True)
loan_test['Gender'].fillna('Male',inplace=True)
loan_test['Married'].fillna('Yes',inplace=True)
loan_test['Dependents'].fillna(0,inplace=True)
loan_test['Self_Employed'].fillna('No',inplace=True)
loan_test['Loan_Amount_Term'].fillna(360,inplace=True)
loan_test['Credit_History'].fillna(1,inplace=True)

loan_test['TotalIncome']=loan_test['ApplicantIncome']+loan_test['CoapplicantIncome']

print('No. of missing values in test')
print(loan.apply(lambda x: sum(x.isnull()),axis=0))

loan_test['LoanAmount_log']=np.log(loan_test['LoanAmount'])
loan_test['TotalIncome_log']=np.log(loan_test['TotalIncome'])

loan_dummy_test=pd.get_dummies(data=loan_test, columns=['Gender','Married','Dependents','Education','Self_Employed','Property_Area'])
print(loan_dummy_test.head(2))
print(loan_dummy_test.columns)



pred1=model1.predict(feat1)

print('Logistic regression metrics:')
print(classification_report(loan_dummy['Loan_Status'],pred1))
print(confusion_matrix(loan_dummy['Loan_Status'],pred1))

model2=LogisticRegression()
model2.fit(feat2,outcome)

pred2=model2.predict(feat2)
print('Logistic Regression metrics on logarithmic converted data:')
print(classification_report(loan_dummy['Loan_Status'],pred2))
print(confusion_matrix(loan_dummy['Loan_Status'],pred2))


model3=LogisticRegression()
model3.fit(feat3,outcome)

pred3=model3.predict(feat3)
print('Logistic Regression metrics on square rooted value:')
print(classification_report(loan_dummy['Loan_Status'],pred3))
print(confusion_matrix(loan_dummy['Loan_Status'],pred3))

#Logistic Regression model on logarithmic converted data is having better results than others


#Decision Tree Classifier

model4=DecisionTreeClassifier(min_samples_split=5,max_depth=10)
model4.fit(feat1,outcome)

pred4=model4.predict(feat1)
print('Decision Tree model metrics:')
print(classification_report(loan_dummy['Loan_Status'],pred4))
print(confusion_matrix(loan_dummy['Loan_Status'],pred4))

model5=DecisionTreeClassifier(min_samples_split=5,max_depth=10)
model5.fit(feat2,outcome)

pred5=model5.predict(feat2)
print('Decision Tree model metrics on log data:')
print(classification_report(loan_dummy['Loan_Status'],pred5))
print(confusion_matrix(loan_dummy['Loan_Status'],pred5))

model6=DecisionTreeClassifier(min_samples_split=5,max_depth=10)
model6.fit(feat3,outcome)

pred6=model6.predict(feat3)
print('Decision Tree model metrics on square rooted data:')
print(classification_report(loan_dummy['Loan_Status'],pred6))
print(confusion_matrix(loan_dummy['Loan_Status'],pred6))


#Random Forest Classifier

model7=RandomForestClassifier(n_estimators=50,min_samples_split=3,max_depth=8)
model7.fit(feat1,outcome)

pred7=model7.predict(feat1)
print('Random Forest model metrics:')
print(classification_report(loan_dummy['Loan_Status'],pred7))
print(confusion_matrix(loan_dummy['Loan_Status'],pred7))

model8=RandomForestClassifier(n_estimators=50,min_samples_split=3,max_depth=8)
model8.fit(feat2,outcome)

pred8=model8.predict(feat2)
print('Random Forest model metrics on log data:')
print(classification_report(loan_dummy['Loan_Status'],pred8))
print(confusion_matrix(loan_dummy['Loan_Status'],pred8))

model9=RandomForestClassifier(n_estimators=50,min_samples_split=3,max_depth=8)
model9.fit(feat3,outcome)

pred9=model9.predict(feat3)
print('Random FOrest model metrics on square rooted data:')
print(classification_report(loan_dummy['Loan_Status'],pred9))
print(confusion_matrix(loan_dummy['Loan_Status'],pred9))

#Let's apply Random Forest model
loan_dummy_test['predicted_output']=model8.predict(loan_dummy_test.drop(['Loan_ID','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','TotalIncome'],axis=1))

print(loan_dummy_test.head())


#Exporting test data
loan_dummy_test.to_csv('output.csv',sep='\t')

#loan_test['pred_output']=model8.predict(loan_dummy_test.drop(['Loan_ID','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','TotalIncome'],axis=1))

#loan_test.drop(['LoanAmount_log','TotalIncome_log']).to_csv('loanout.csv',sep='\t')



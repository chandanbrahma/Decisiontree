
###importing the dataset
import pandas as pd

data = pd.read_csv("E:\\assignment\\decisiontree\\Fraud_Check.csv")
data.head()
data.describe()
data.info()

##lets convert all  the non catagory columns into catagory format and replacing them into the dataframe
##catagorising the income column as >30000 is good and less than 30000 as risky income
data['income'].describe()
income_category= pd.cut(data.income,bins=[0,30000,100000],labels=['risky','good'])

##inserting the column
data.insert(0,'taxable_income',income_category)
data.taxable_income.value_counts()

## catagorising population column
data.population.describe()
population_catagory=pd.cut(data.population,bins=[0,67000,120000,200000],labels=['low','medium','high'])

##inserting the column
data.insert(3,'population_catagory',population_catagory)
data.population_catagory.value_counts()


##catagorising expperience column
data.experience.describe()
experience_catagory=pd.cut(data.experience,bins=[-1,8,20,30],labels=['low','medium','high'])
data.insert(4,'experience_catagory',experience_catagory)
experience_catagory.describe()

## now droping the catagorisd columns
data.drop(columns=['income','population','experience'],inplace=True)
data.info()


##as we have created a dataframe that contains only catagorical variables.So now lets convert the strings into respective intcodes using pandas.
data['taxable_income'],_ = pd.factorize(data['taxable_income'])
data['Marital.Status'],_ = pd.factorize(data['Marital.Status'])
data['population_catagory'],_ = pd.factorize(data['population_catagory'])
data['Undergrad'],_ = pd.factorize(data['Undergrad'])
data['Urban'],_ = pd.factorize(data['Urban'])
data['experience_catagory'],_ = pd.factorize(data['experience_catagory'])

data.head()
data.info()##everything are converted into integer form

###selecting the target as y and predictor as x
x=data.iloc[:,1:]
y=data.iloc[:,0]

##spliting data randomly into 80% training and 20% test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


###now creating and training the desicion tree
from sklearn.tree import  DecisionTreeClassifier
from sklearn import metrics
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(x_train, y_train)


##using the model to make predictions with the test data
y_pred = dtree.predict(x_test)



##checking the perfermnce of our model
count_misclassified = (y_test != y_pred).sum()
print(count_misclassified)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

##so our algorithem is capable of proving a model with ~80% accuracy and with a misclassified values of only  25.








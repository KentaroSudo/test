#########################################
#Data Scientist Track 1 Hands-on Interview
##########################################


# Full interview script in Box at:
# https://thinkbiganalytics.app.box.com/notes/36937708426


##########################################
# Module 1: LOAD DATA
##########################################

# Load the data into R/Python and take a quick look to make sure it was loaded correctly.
import pandas as pd
data = pd.DataFrame.from_csv('/path/to/track_1.dataset',index_col=False)

data.describe()
data
data.head()
data.shape
data.columns
data['accepted'].describe()


##########################################
# Module 2: DATA EXPLORATION
##########################################

# Whether an application was accepted is coded in a column named "accepted" which has only two levels: 1=rejected and 2=accepted.  
# We would like to alter this variable in place so that 0=rejected and 1=accepted.  

data['accepted'].head() # look at first few values
data['accepted'] = data['accepted'] -1


# What percentage of the loan applications were accepted?
# 700/1000 = 70%

float(data['accepted'].sum())/float(data['accepted'].count())

# What is the most common purpose for the loan applications?
# radio/television 280

data['purpose'].value_counts()

# What is the average age of loan applicants that were accepted compared to the average age of applicants that were rejected?

data.groupby('accepted')['age'].mean()

##########################################
# Module 3: DATA WRANGLING
##########################################

# The "existing_loans" field appears to have an issue with some preceding junk characters we need to clean up.  
# In each entry of the field there are 0 to 3 lowercase alphabet characters which precede the true numeric value for the number of existing loans.  
# Please remove these preceding characters

data['existing_loans'].head() # Look at the first few values

# via string methods regular expression replacement

data['existing_loans'] = data['existing_loans'].str.replace('^[a-z]{0,3}','')

##########################################
# Module 4: PREDICTIVE MODELING
##########################################

# Next we want to build a model to predict the outcome of each loan application.  
# However, for legal reasons we are only allowed to use the following 5 features: age, checking_status, duration, job, and purpose.
# Please build a model to predict the outcome of each loan application.
# Feel free to use the modeling technique of your choice. 

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
model = LinearRegression()
le_CS = preprocessing.LabelEncoder()
data['CS_encoded'] = le_CS.fit_transform(data['checking_status'])
cols = ['age','CS_encoded','installment_commitment','duration','loan_amount']
myX = data[cols]
myY = data['accepted']
model.fit(myX,myY)


# Now we want to use the model you just built to make predictions for the entire dataset.
# If your model creates continuous predictions please convert those to binary 0/1 predictions.  
# What percent of loan applications did your model predict correctly?

pd.crosstab(model.predict(data[cols]).round(0),data['accepted'])

# What are some other quantitative measures to gauge how accurate those predictions are?

import sklearn.metrics as mx # via sklearn metrics package

# recall/sensitivity = true positive rate = fraction of positive instances that are correctly identified as such

mx.recall_score(data['accepted'],model.predict(data[cols]).round(0))

# precision/specificity = true negative rate = fraction of negative instances that are correctly identified as such

mx.precision_score(data['accepted'],model.predict(data[cols]).round(0))

# What are some ways that the existing model that you built could be improved
# Variety of options depending on model choice. Observe & ensure candidate chooses & acts upon a model improvement strategy


##########################################
# Module 5: HYPOTHESIS TESTING
##########################################

# Please take a second to look through this code..
# explain what the code is doing..
# what type of question would this piece of code help to answer?

import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from statistics import mean, stdev
from math import exp

data = pd.read_csv("loan_data_track1.csv")
numeric_model_fields = ["age", "duration", "loan_amount", "installment_commitment"]
X = data[numeric_model_fields]
X.insert(len(X.columns), 'own_telephone_Yes', pd.get_dummies(data['own_telephone'])['Yes'])
X_col_list = list(X.columns)
y = data["accepted"]
logistic_model = LogisticRegression()
sample_size = 0.6
result_list = []
for i in range(100):
    sample_index = random.sample(range(len(data)),int(len(data) * sample_size))
    model = logistic_model.fit(X.ix[sample_index], y.ix[sample_index])
    result_list.append(exp(model.coef_[0][X_col_list.index('own_telephone_Yes')]))
    
print(mean(result_list))
print(stdev(result_list))
plt.hist(result_list)
plt.show()


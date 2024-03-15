
# Load and separate data into independent and dependent variables
import pandas as pd

# Load data.
# Last column is the dependent variable. We want to use the four independent variables to make a prediction on Log(S)
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv') # data frame
# Separate data frame into X and y variables
y = df['logS']
X = df.drop('logS', axis=1) # remove y variable. axis=1 means it separates by column, axis=0 would be row


# Split data set into training and testing set
from sklearn.model_selection import train_test_split
# training set has 80% of data, test set has 20% of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100) # random state is assigned specific number so that every time it is run we get the same split


# Model building

# Linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression() # Linear regression model 
lr.fit(X_train, y_train) # Train the linear regression model on the training data set





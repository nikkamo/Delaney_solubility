
################    Load and separate data into independent and dependent variables     ################
import pandas as pd

# Load data.
# Last column is the dependent variable. We want to use the four independent variables to make a prediction on Log(S)
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv') # data frame
# Separate data frame into X and y variables
y = df['logS']
X = df.drop('logS', axis=1) # remove y variable. axis=1 means it separates by column, axis=0 would be row


################    Split data set into training and testing set    ################     
from sklearn.model_selection import train_test_split
# training set has 80% of data, test set has 20% of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100) # random state is assigned specific number so that every time it is run we get the same split


################    Building the liner regression model  ################
# Training linear regression model
from sklearn.linear_model import LinearRegression

lr = LinearRegression() # Linear regression model 
lr.fit(X_train, y_train) # Train the linear regression model on the training data set

# Applying linear regression model to make prediction on training and testing set
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)


# Evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score

# Compare actual values with predicted values for both training and testing set
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Create pandas data frame to show the results in a tidy way
lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']


################    Building the Random Forest model  ################
# Training random forest model
from sklearn.ensemble import RandomForestRegressor # y-variable is quantitative so we build regression model. If it was categorial we build classification model. So we use regressor instead of classifier

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train,y_train)

# Applying linear regression model to make prediction on training and testing set
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)


# Evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score

# Compare actual values with predicted values for both training and testing set
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

# Create pandas data frame to show the results in a tidy way
rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']


################    Compare models  ################
# combine the two results table into one
df_models = pd.concat([lr_results, rf_results], axis=0) # axis=0 because we want to combine in a row-wise manner
df_models.reset_index(drop=True) # to the left there's index 0 for both tables, reset these and drop the title on the column
print(df_models)


################    Data visualization of prediction results  ################
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c='green', alpha=0.3) # alpha adjustes darkness of the samples - regions that have many points are darker than regions with less points

z = np.polyfit(y_train, y_lr_train_pred, 1) # make a trend line
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='red')

plt.xlabel('Experimental Log(S)')
plt.ylabel('Predicted Log(S) (linear model)')
plt.show()
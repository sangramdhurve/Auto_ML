#  Importing useful libraries for function
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor


#  For reading our data set(defining a function)
def load_data_from_path(filepath):
    df = pd.read_csv(filepath)
    return df


#  For splitting our dataset into Test and Train in Target and features columns
def split_train_test_data():
    df = load_data_from_path("The path of the file or csv")
    X = df.drop(columns=[-1])  # setting target column.
    y = df[-1]  # setting independent variables or features.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


#  Form Sklearn we can call any model, I'm using only one example here.
def linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr


#  After Training the model we need prediction file to seen predicted of model.
def predict_on_test_data(modelName, X_test):  # we can write name instead of modelName
    y_test = modelName.predict(X_test)
    filename = str(modelName.__class__.__name__)+"_predicted_output.csv"  # This will return only name of the model
    prediction = pd.DataFrame(y_test)
    pd.DataFrame(y_test).to_csv("data/model_testing_data/"+filename)
    return prediction


# For finding the current working directory so that we can get our predicted save files.
print(':-) This is your file directory:-==> ', os.getcwd())


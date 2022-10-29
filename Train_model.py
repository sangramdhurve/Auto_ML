from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


class Model:
    def linear_regression(self, xtrain, ytrain):
        lr = LinearRegression()
        lr.fit(xtrain, ytrain)
        return lr

    def k_neighbours(self, xtrain, ytrain):
        knn = KNeighborsRegressor()
        knn.fit(xtrain, ytrain)
        return knn

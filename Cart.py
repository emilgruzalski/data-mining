import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from math import sqrt

class Cart:
    params = {  'max_depth' : range(1,10),
                'min_samples_split' : [15, 20, 30, 40, 50],
                'min_samples_leaf' : [5, 10 ,15],
                'max_leaf_nodes': [250]}

    def __init__(self, X_train, X_test, y_train, y_test, seed):
        self.model = GridSearchCV(DecisionTreeRegressor(random_state=seed), self.params, n_jobs=-1)
        self.model.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_pred = self.model.predict(X_train)
        self.y_test_pred = self.model.predict(X_test)

    def model_raitings(self):
        print("Użyte parametry: ", self.model.best_params_)
        print("Zbiór uczący: ")
        self.calc_errors(self.y_train, self.y_train_pred)
        print("Zbiór testowy: ")
        self.calc_errors(self.y_test, self.y_test_pred)
        print("Rozmiar drzewa: ", self.model.best_estimator_.tree_.node_count)
        plt.scatter(range(0,77), self.y_test, color='black', label='Obserwowane')
        plt.scatter(range(0,77), self.y_test_pred, color='blue', label='Przewidywane')
        plt.ylabel('MPG')
        plt.xlabel('Indeks')
        plt.legend()
        plt.show()

    def calc_errors(self, y_true, y_pred):
        print("RMSE: ", round(sqrt(mean_squared_error(y_true, y_pred)), 4))
        print("MAE: ", round(mean_absolute_error(y_true, y_pred)), 4)
        print("MAPE: ", round(100*mean_squared_error(y_true, y_pred)), 4)

    def variables_hierarchy(self):
        tree = self.model.best_estimator_
        hierarchy = pd.Series(tree.feature_importances_, index=self.X_train.columns)
        hierarchy.sort_values(inplace=True)
        hierarchy.iloc[-10:].plot(kind='barh', figsize=(6,4))

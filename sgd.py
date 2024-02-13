from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd

class SGD:
    params = {
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': [0.001, 0.005, 0.01],
        'max_iter': [1000, 3000, 5000],
        'tol': [0.001, 0.01, 0.025],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'l1_ratio': [0.15, 0.5, 0.85]
    }

    def __init__(self, X_train, X_test, y_train, y_test, seed):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Zapisujemy nazwy kolumn
        self.feature_names = X_train.columns

        self.model = GridSearchCV(SGDRegressor(random_state=seed), self.params, n_jobs=-1, cv=5)
        self.model.fit(X_train_scaled, y_train)
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_pred = self.model.predict(X_train_scaled)
        self.y_test_pred = self.model.predict(X_test_scaled)

        print("Użyte parametry: ", self.model.best_params_)

        self.evaluate_model(y_train, self.y_train_pred, 'Zbiór uczący')
        self.evaluate_model(y_test, self.y_test_pred, 'Zbiór testowy')

    def evaluate_model(self, y_true, y_pred, dataset_name):
        print(f"{dataset_name}:")
        print("RMSE: ", round(sqrt(mean_squared_error(y_true, y_pred)), 4))
        print("MAE: ", round(mean_absolute_error(y_true, y_pred), 4))
        print("MAPE: ", round(100 * mean_absolute_error(y_true, y_pred) / y_true.mean(), 4))
        print("R^2: ", round(r2_score(y_true, y_pred), 4))

    def plot_predictions(self):
        # Używanie numerów jako indeksów na osi X
        num_samples = len(self.y_test)
        plt.figure(figsize=(10, 6))  # Dostosuj rozmiar wykresu według potrzeb
        plt.scatter(range(num_samples), self.y_test, color='black', label='Obserwowane')
        plt.scatter(range(num_samples), self.y_test_pred, color='blue', label='Przewidywane')
        plt.ylabel('MPG')
        plt.xlabel('Indeks')
        plt.legend()
        plt.show()

    def feature_importance(self):
        best_model = self.model.best_estimator_
        importance = pd.Series(best_model.coef_, index=self.feature_names)
        importance.sort_values(inplace=True)
        importance.plot(kind='barh', figsize=(6, 4))
        plt.title("Ważność cech")
        plt.show()

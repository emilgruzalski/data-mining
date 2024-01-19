import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
import pprint
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych
file_path = 'red_wine.csv'
df = pd.read_csv(file_path, sep=';', decimal=',')

# Podział danych na zbiór uczący i testowy
train_data, test_data = train_test_split(df, test_size=0.2, random_state=308303)

# Przygotowanie danych
X_train = train_data.drop('quality', axis=1)
y_train = train_data['quality']
X_test = test_data.drop('quality', axis=1)
y_test = test_data['quality']

# Definiowanie siatki parametrów do przetestowania
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Inicjalizacja Grid Search z modelem RandomForestClassifier
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=308303), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Przeprowadzenie doboru hiperparametrów
grid_search.fit(X_train, y_train)

# Najlepszy model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Ocena najlepszego modelu na zbiorze testowym
y_pred_classification = best_model.predict(X_test)

# Inicjalizacja Grid Search z modelem RandomForestRegressor
grid_search_regression = GridSearchCV(
    estimator=RandomForestRegressor(random_state=308303),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Przeprowadzenie doboru hiperparametrów
grid_search_regression.fit(X_train, y_train)

# Najlepszy model
best_model_regression = grid_search_regression.best_estimator_
best_params_regression = grid_search_regression.best_params_

# Ocena najlepszego modelu na zbiorze testowym
y_pred_regression = best_model_regression.predict(X_test)
y_pred_regression_rounded = np.round(y_pred_regression)

# Ocena modeli
accuracy_classification = accuracy_score(y_test, y_pred_classification)
accuracy_regression = accuracy_score(y_test, y_pred_regression_rounded)
mae_classification = mean_absolute_error(y_test, y_pred_classification)
mae_regression = mean_absolute_error(y_test, y_pred_regression_rounded)

# Trafność z dopuszczalnym odstępstwem o 1
accuracy_classification_with_deviation = accuracy_score(y_test, y_pred_classification, normalize=False)
for i, (true, pred) in enumerate(zip(y_test, y_pred_classification)):
    if abs(true - pred) == 1:
        accuracy_classification_with_deviation += 1
accuracy_classification_with_deviation = accuracy_classification_with_deviation / len(y_test)

accuracy_regression_with_deviation = accuracy_score(y_test, y_pred_regression_rounded, normalize=False)
for i, (true, pred) in enumerate(zip(y_test, y_pred_regression_rounded)):
    if abs(true - pred) == 1:
        accuracy_regression_with_deviation += 1
accuracy_regression_with_deviation = accuracy_regression_with_deviation / len(y_test)

# Ważność cech
feature_importances_classification = best_model.feature_importances_
feature_importances_regression = best_model_regression.feature_importances_

# Utworzenie modelu KMeans
kmeans = KMeans(n_clusters=8, random_state=308303, n_init='auto')
kmeans.fit(X_train)

# Przypisanie grup do danych uczących
train_clusters = kmeans.predict(X_train)

# Opis profilu grup
cluster_profiles = X_train.groupby(train_clusters).mean()

# Przypisanie grup do danych testowych
test_clusters = kmeans.predict(X_test)

# Sprawdzenie związku grup z jakością wina w zbiorze testowym
test_data_with_clusters = test_data.copy()
test_data_with_clusters['Cluster'] = test_clusters
cluster_quality_relation = test_data_with_clusters.groupby('Cluster')['quality'].mean()

# Analiza rozkładu grup w zbiorach uczącym i testowym
train_cluster_counts = pd.Series(train_clusters).value_counts()
test_cluster_counts = pd.Series(test_clusters).value_counts()

print("Liczba przykładów w klastrach w zbiorze uczącym:")
print(train_cluster_counts)
print("\nLiczba przykładów w klastrach w zbiorze testowym:")
print(test_cluster_counts)

# Wizualizacja profilu klastrów
for i in range(8):
    cluster_data = X_train[train_clusters == i]
    cluster_data.mean().plot(kind='bar')
    plt.title(f'Profil klastra {i}')
    plt.show()

# Wyniki
print('Model klasyfikacji:')
print('Użyte parametry:', best_params)
print('Ważność cech:')
pprint.pprint(list(zip(X_train.columns, feature_importances_classification)))
print('Trafność:', accuracy_classification)
print('Trafność z dopuszczalnym odstępstwem o 1:', accuracy_classification_with_deviation)
print('Średni błąd bezwzględny (MAE):', mae_classification)
print('\nModel regresji:')
print('Użyte parametry:', best_params_regression)
print('Ważność cech:')
pprint.pprint(list(zip(X_train.columns, feature_importances_regression)))
print('Trafność:', accuracy_regression)
print('Trafność z dopuszczalnym odstępstwem o 1:', accuracy_regression_with_deviation)
print('Średni błąd bezwzględny (MAE):', mae_regression)
print('\nProfile otrzymanych grup:\n', cluster_profiles)
print('Związek grup z jakością wina:\n', cluster_quality_relation)
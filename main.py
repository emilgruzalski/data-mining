import pandas as pd
from sklearn.model_selection import train_test_split
from mlp import MLPClassifier, evaluate_model
from numpy.random import seed

# Wczytanie danych
data = pd.read_csv('ścieżka_do_pliku.csv')

# Wyspecyfikowanie zmiennych
data['Sex'] = data['Sex'].astype('category')
data['ChestPainType'] = data['ChestPainType'].astype('category')
data['RestingECG'] = data['RestingECG'].astype('category')
data['ExerciseAngina'] = data['ExerciseAngina'].astype('category')
data['ST_Slope'] = data['ST_Slope'].astype('category')
data['HeartDisease'] = data['HeartDisease'].astype('int')

# Ustawienie ziarna generatora liczb losowych
seed_value = int(data.index.mean())
seed(seed_value)

# Podział danych na zbiory uczący i testowy
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

# Trenowanie modelu MLP
mlp_model = MLPClassifier()
mlp_model.fit(X_train, y_train)

# Ewaluacja modelu
evaluate_model(mlp_model, X_test, y_test)

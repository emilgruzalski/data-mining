import pandas as pd
from sklearn.model_selection import train_test_split
from mlp import MLPClassifier, evaluate_model as evaluate_mlp
from extra_trees import ExtraTreesClassifierModel, evaluate_model as evaluate_extra_trees

# Wczytanie danych
data = pd.read_csv('heart.csv')

# Przetwarzanie danych
data['Sex'] = data['Sex'].astype('category')
data['ChestPainType'] = data['ChestPainType'].astype('category')
# Usunięcie kolumny 'RestingECG'
data = data.drop('RestingECG', axis=1)
data['ExerciseAngina'] = data['ExerciseAngina'].astype('category')
data['ST_Slope'] = data['ST_Slope'].astype('category')

# Konwersja danych kategorialnych na numeryczne
data = pd.get_dummies(data, drop_first=True)

# Podział danych na cechy i etykiety
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Podział danych na zbiory uczące i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu MLP
mlp_model = MLPClassifier()
mlp_model.fit(X_train, y_train)

# Ocena modelu MLP
print("Wyniki dla MLP:")
evaluate_mlp(mlp_model, X_test, y_test)
print("\n")

# Trenowanie modelu Extra Trees
et_model = ExtraTreesClassifierModel()
et_model.fit(X_train, y_train)

# Ocena modelu Extra Trees
print("Wyniki dla Extra Trees:")
evaluate_extra_trees(et_model, X_test, y_test)

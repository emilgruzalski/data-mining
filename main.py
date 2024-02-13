import pandas as pd
from sklearn.model_selection import train_test_split
from cart import Cart
from sgd import SGD

# Load, set and remove lack of data
cars = pd.read_csv("auto_mpg.csv", sep=";", na_values=' ')
cars.dropna(how="any", inplace=True)

index_values = cars.index.to_numpy()

cars.displacement = cars.displacement.str.replace(',', '.').astype(float)
cars.acceleration = cars.acceleration.str.replace(',', '.').astype(float)
cars.mpg = cars.mpg.str.replace(',', '.').astype(float)
cars.set_index('car_name', inplace=True)

# Set seed and split data for two sets
seed_value = int(index_values.mean())
print(f"Seed value: {seed_value}")

X = cars.drop('mpg', axis=1)
y = cars.mpg

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

cart_model = Cart(X_train, X_test, y_train, y_test, seed_value)
cart_model.model_raitings()
cart_model.variables_hierarchy()

sgd_model = SGD(X_train, X_test, y_train, y_test, seed_value)
sgd_model.plot_predictions()
sgd_model.feature_importance()
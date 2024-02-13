import pandas as pd
from sklearn.model_selection import train_test_split
from src.mlp import MLPClassifier, evaluate_model as evaluate_mlp
from src.extra_trees import ExtraTreesClassifierModel, evaluate_model as evaluate_extra_trees

# Load data
data = pd.read_csv('../data/heart.csv')

# Set the seed for the random number generator
index_values = data.index.to_numpy()
seed_value = int(index_values.mean())
print(f"Seed value: {seed_value}")

# Data processing
data['Sex'] = data['Sex'].astype('category')
data['ChestPainType'] = data['ChestPainType'].astype('category')
# Removing the 'RestingECG' column
data = data.drop('RestingECG', axis=1)
# Removing the row where the human's blood pressure was 0, as it indicates that the person is deceased.
data = data[data.RestingBP != 0]
data['ExerciseAngina'] = data['ExerciseAngina'].astype('category')
data['ST_Slope'] = data['ST_Slope'].astype('category')

# Convert categorical data to numerical
data = pd.get_dummies(data, drop_first=True)

# Split data into features and labels
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

# Train the MLP model
mlp_model = MLPClassifier(random_state=seed_value)
mlp_model.fit(X_train, y_train)

# Evaluate the MLP model
print("Results for MLP:")
evaluate_mlp(mlp_model, X_test, y_test)
print("\n")

# Train the Extra Trees model
et_model = ExtraTreesClassifierModel(random_state=seed_value)
et_model.fit(X_train, y_train)

# Evaluate the Extra Trees model
print("Results for Extra Trees:")
evaluate_extra_trees(et_model, X_test, y_test)
print("\n")

# Feature importances in the Extra Trees model
importances = et_model.feature_importances()
print("Feature importances in the Extra Trees model:")
for name, importance in zip(X.columns, importances):
    print(f"{name}: {importance}")

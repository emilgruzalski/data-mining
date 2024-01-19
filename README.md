# Red Wine Quality Analysis

## Description
This repository hosts a project focused on analyzing red wine quality using machine learning techniques. The project leverages models like Random Forest Classifier and Regressor, and KMeans clustering, providing insights into factors influencing wine quality.

## Features
- Loading data from the CSV file.
- Splitting data into training and test sets.
- Implementing Random Forest models using `sklearn`'s `RandomForestClassifier` and `RandomForestRegressor`.
- KMeans clustering for data segmentation.
- Model evaluation with accuracy score, mean absolute error, and other relevant metrics.
- Data visualization using matplotlib and seaborn for insightful EDA.
- GridSearchCV for hyperparameter tuning in models.

## Data Categories
The project analyzes the following attributes from the red wine dataset:
1. **Fixed Acidity** [Numerical]
2. **Volatile Acidity** [Numerical]
3. **Citric Acid** [Numerical]
4. **Residual Sugar** [Numerical]
5. **Chlorides** [Numerical]
6. **Free Sulfur Dioxide** [Numerical]
7. **Total Sulfur Dioxide** [Numerical]
8. **Density** [Numerical]
9. **pH** [Numerical]
10. **Sulphates** [Numerical]
11. **Alcohol** [Numerical]
12. **Quality** (Rating) [Categorical]

## Requirements
- Python 3.x
- `pandas`
- `sklearn`
- `matplotlib`

## Installation
Ensure you have Python installed along with the libraries mentioned above. If not, you can install them using the following command:
```bash
pip install pandas scikit-learn matplotlib
```

## Usage
To run the project, clone the repository and execute the `red_wine.py` file:
```bash
git clone https://github.com/emilgruzalski/red-wine.git
cd red-wine
python red_wine.py
```

## License
This project is released under the MIT License.

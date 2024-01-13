# Auto MPG Analysis

## Description
This repository contains a project focused on analyzing the Auto MPG (Miles Per Gallon) data using machine learning techniques. The project employs two models: Stochastic Gradient Descent (SGD) and Decision Tree Regressor (CART).

## Features
- Load data from a CSV file
- Split data into training and test sets
- Implementation of the SGD model using `sklearn`'s `SGDRegressor`.
- Implementation of the CART model using `sklearn`'s `DecisionTreeRegressor`.
- Model evaluation based on metrics such as Mean Squared Error, Mean Absolute Error, and R2 Score.
- Cross-validation for hyperparameter tuning.

## Data Categories
The project considers the following categories from the auto mpg dataset:
1. **Cylinders**: number of cylinders [Numerical value]
2. **Displacement**: engine displacement [cc]
3. **Horsepower**: engine power [HP]
4. **Weight**: vehicle weight [kg]
5. **Acceleration**: acceleration [seconds to reach 60 mph]
6. **ModelYear**: model year of the vehicle [Year]
7. **Origin**: origin of the vehicle [1: USA, 2: Europe, 3: Asia]
8. **CarName**: name of the car [Text]
9. **MPG**: fuel consumption [miles per gallon]

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
To run the project, clone the repository and execute the `main.py` file:
```bash
git clone https://github.com/emilgruzalski/auto-mpg.git
cd auto-mpg
python main.py
```

## License
This project is released under the MIT License.

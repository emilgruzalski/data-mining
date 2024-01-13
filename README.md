# Heart Disease Analysis

## Description
This repository contains a project for analyzing heart disease data using machine learning techniques. The project utilizes two models: a Multilayer Perceptron Neural Network (MLP) and an Extra Trees Classifier.

## Features
- Load data from a CSV file
- Split data into training and test sets
- Implementation of the MLP model from the `sklearn` library
- Implementation of the Extra Trees Classifier model
- Evaluation of models based on various metrics

## Data Categories
The project considers the following categories from the heart disease dataset:
1. **Age**: age of the patient [years]
2. **Sex**: sex of the patient [M: Male, F: Female]
3. **ChestPainType**: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4. **RestingBP**: resting blood pressure [mm Hg]
5. **Cholesterol**: serum cholesterol [mm/dl]
6. **FastingBS**: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
7. **RestingECG**: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
8. **MaxHR**: maximum heart rate achieved [Numeric value between 60 and 202]
9. **ExerciseAngina**: exercise-induced angina [Y: Yes, N: No]
10. **Oldpeak**: oldpeak = ST [Numeric value measured in depression]
11. **ST_Slope**: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
12. **HeartDisease**: output class [1: heart disease, 0: Normal]

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
git clone https://github.com/emilgruzalski/heart-disease.git
cd heart-disease
python main.py
```

## License
This project is released under the MIT License.

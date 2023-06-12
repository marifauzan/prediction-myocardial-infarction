# Importing necessary libraries
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create a new Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

# Define a route for the API that accepts a POST request with a JSON payload
@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model based on user selection
    model_name = request.form.get('ModelType')  # Get the selected model name from the form
    if model_name not in ['Stacked Generalization', 'CatBoost', 'LGBM', 'XGBoost', 'GBM', 'AdaBoost', 'Extra Trees', 'Random Forest', 'Bagging']:
        return jsonify({'error': 'Invalid model selection'}), 400
    
    # Load the selected model
    model = joblib.load(f'./model/{model_name}.pkl')
    
    # Get the form data
    age = float(request.form.get('Age'))
    sex = float(request.form.get('Sex'))
    cp = float(request.form.get('ChestPainType'))
    trestbps = float(request.form.get('RestingBP'))
    chol = float(request.form.get('Cholesterol'))
    fbs = float(request.form.get('FastingBS'))
    restecg = float(request.form.get('RestingECG'))
    thalach = float(request.form.get('MaxHR'))
    exang = float(request.form.get('ExerciseAngina'))
    oldpeak = float(request.form.get('Oldpeak'))
    stslope = float(request.form.get('ST_Slope'))
    
    # Create a pandas dataframe with the input data
    input_df = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [cp],
        'RestingBP': [trestbps],
        'Cholesterol': [chol],
        'FastingBS': [fbs],
        'RestingECG': [restecg],
        'MaxHR': [thalach],
        'ExerciseAngina': [exang],
        'Oldpeak': [oldpeak],
        'ST_Slope': [stslope]
    })
    
    # Make the prediction
    prediction = model.predict(input_df)
    
    # Render the template with the prediction result
    return render_template('form.html', prediction=prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
    
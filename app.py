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
    return render_template('index.html')

@app.route('/form-upload')
def formUpload():
    return render_template('upload.html')

@app.route('/form')
def form():
    return render_template('form.html')

# Define a route for the API that accepts a POST request with a JSON payload
@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model based on user selection
    model_name = request.form.get('ModelType')  # Get the selected model name from the form
    if model_name not in ['Stacked Generalization', 'GBM', 'Random Forest']:
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

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Mengunggah file CSV
        file = request.files['file']
        df = pd.read_csv(file)
        df_copy = df.copy()

        # Memilih model prediksi yang dipilih oleh pengguna
        model = request.form['model']

        # Melakukan prediksi menggunakan model yang dipilih
        predictions = predictCSV(df, model)

        # Menghitung metrik evaluasi
        target = df['condition']  # Kolom target pada dataset
        accuracy = accuracy_score(target, predictions['Prediction'])
        precision = precision_score(target, predictions['Prediction'])
        sensitivity = recall_score(target, predictions['Prediction'])
        f1 = f1_score(target, predictions['Prediction'])

        # Format metrics as percentages
        accuracy = '{:.2%}'.format(accuracy)
        precision = '{:.2%}'.format(precision)
        sensitivity = '{:.2%}'.format(sensitivity)
        f1 = '{:.2%}'.format(f1)

        # Menggabungkan hasil prediksi dan metrik evaluasi dalam DataFrame
        result_df = pd.DataFrame({'Age': df_copy['Age'], 'Sex': df_copy['Sex'], 'ChestPainType': df_copy['ChestPainType'],
                                   'RestingBP': df_copy['RestingBP'], 'Cholesterol': df_copy['Cholesterol'], 'FastingBS': df_copy['FastingBS'], 
                                   'RestingECG': df_copy['RestingECG'], 'MaxHR': df_copy['MaxHR'], 'ExerciseAngina': df_copy['ExerciseAngina'],
                                    'Oldpeak': df_copy['Oldpeak'], 'ST_Slope': df_copy['ST_Slope'], 'Prediction': predictions['Prediction'], 'Target': target})
        result_df = result_df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'Prediction', 'Target']]  # Mengurutkan kolom

        # Menampilkan tabel hasil prediksi dan metrik evaluasi
        return render_template('result.html', result=result_df, accuracy=accuracy, precision=precision, sensitivity=sensitivity, f1=f1)
        
    return render_template('upload.html')

def preprocess_data(df):
    # Define a dictionary to map
    sex_map = {'M': 0, 'F': 1}
    ch_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
    restecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
    exang_map = {'N': 0, 'Y': 1}
    slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}

    # Use the map() method to replace the values
    df['Sex'] = df['Sex'].map(sex_map)
    df['ChestPainType'] = df['ChestPainType'].map(ch_map)
    df['RestingECG'] = df['RestingECG'].map(restecg_map)
    df['ExerciseAngina'] = df['ExerciseAngina'].map(exang_map)
    df['ST_Slope'] = df['ST_Slope'].map(slope_map)

    # Specify the column names you want to scale
    columns_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

    # Create an instance of the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the data and perform the transformation
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    df.rename(columns={'HeartDisease': 'condition'}, inplace=True)

    return df

def predictCSV(df, model):
    # Load model dari file .pkl
    loaded_model = joblib.load(f'./model/{model}.pkl')

    # Preprocess data
    preprocessed_df = preprocess_data(df)

    # Menentukan fitur (X)
    X = preprocessed_df.drop('condition', axis=1)

    # Melakukan prediksi
    predictions = loaded_model.predict(X)

    result_df = preprocessed_df.copy()
    result_df['Prediction'] = predictions

    return result_df


if __name__ == '__main__':
    app.run(debug=True)
    
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Directories to store uploaded data and model
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

model = None
scaler = None


# Helper function to train the machine learning model
def train_model(df, model_type='logistic'):
    # Preprocessing: Drop rows with missing values
    df = df.dropna()

    # Feature selection (X) and target variable (y)
    X = df[['Run_Time', 'Temperature']]  # Features
    y = df['Downtime_Flag']  # Target variable

    # Split data into training and testing (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    global scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose model type: Logistic Regression or Decision Tree
    if model_type == 'logistic':
        model = LogisticRegression()
    else:
        model = DecisionTreeClassifier(random_state=42)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test_scaled)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save the trained model to disk
    joblib.dump(model, os.path.join(app.config['MODEL_FOLDER'], 'model.pkl'))

    return model, accuracy, f1


@app.route("/", methods=["GET"])
def index():
    return  render_template('index.html')

# Upload endpoint: Accept CSV file for training data
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csvfile' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['csvfile']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Load the uploaded CSV data into DataFrame
        df = pd.read_csv(filename)

        # Save the dataset for later use in training
        df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'manufacturing_data.csv'), index=False)

        # Train the model using the uploaded data
        global model
        model, accuracy, f1 = train_model(df, model_type='logistic')  # Logistic regression

        return jsonify({
            'message': 'File uploaded and model trained successfully',
            'accuracy': accuracy,
            'f1_score': f1
        }), 200

    return jsonify({'error': 'Invalid file format'}), 400


# Train endpoint: Train the model and return performance metrics
@app.route('/train', methods=['POST'])
def train():
    # Check if the dataset exists
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], 'manufacturing_data.csv')
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'No data uploaded yet. Please upload a dataset first.'}), 400

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Train the model and evaluate performance
    global model
    model, accuracy, f1 = train_model(df, model_type='logistic')  # Logistic regression

    # Return the performance metrics
    return jsonify({
        'accuracy': accuracy,
        'f1_score': f1
    }), 200


# Predict endpoint: Make predictions for new data
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 400

    data = request.get_json()

    # Extract features for prediction
    run_time = data.get('Run_Time')
    temperature = data.get('Temperature')

    if run_time is None or temperature is None:
        return jsonify({'error': 'Missing required fields: Run_Time, Temperature'}), 400

    # Prepare data for prediction
    X_new = [[run_time, temperature]]
    X_new_scaled = scaler.transform(X_new)

    # Make prediction
    prediction = model.predict(X_new_scaled)
    confidence = model.predict_proba(X_new_scaled)[0][prediction[0]]  # Get prediction confidence

    downtime_status = 'Yes' if prediction[0] == 1 else 'No'

    return jsonify({'Downtime': downtime_status, 'Confidence': confidence}), 200


if __name__ == '__main__':
    app.run(debug=True)

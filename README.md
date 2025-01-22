# Manufacturing_Data_analysis_flask(Prediction)



This is a simple RESTful API built using Flask and scikit-learn to predict machine downtime or product defects based on manufacturing data. The API allows you to upload a CSV dataset, train a machine learning model, and make predictions on new data. 

## Features

- **File Upload**: Upload a CSV file containing manufacturing data.
- **Model Training**: Train a machine learning model on the uploaded data.
- **Prediction**: Use the trained model to predict machine downtime based on features like `Temperature` and `Run_Time`.
- **Performance Metrics**: View model performance metrics like **accuracy** and **F1-score**.

## Prerequisites

Before running the project, make sure you have the following installed:

- Python 3.7+
- `pip` (Python package installer)

## Setup Instructions

### 1. Clone the repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/AmolK961/Manufacturing_Data_analysis_flask.git
cd Manufacturing_Data_analysis_flask



2. Create a virtual environment (optional but recommended)
Create a virtual environment to manage dependencies for this project:
python -m venv venv
Activate the virtual environment:
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Start the Flask server
python app.py

5. Testing the API
a. Upload a CSV file


and test all endpoint .

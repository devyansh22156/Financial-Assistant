from flask import Flask, render_template, request
import pickle  # To load the trained model
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load the trained model (e.g., XGBoost model from your workflow)
model_path = "archive/models/xgboost_model.pkl"  # Replace with the actual path to your model file
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define columns (Ensure this matches your dataset)
input_columns = [
    'Age', 'Income', 'Credit Score', 'Loan Amount',
    'Years at Current Job', 'Debt-to-Income Ratio',
    'Assets Value', 'Number of Dependents', 'Previous Defaults',
    'Gender', 'Marital Status', 'Loan Purpose', 'Education Level',
    'Employment Status', 'Marital Status Change', 'Payment History'
]

# Categorical mappings
categorical_mappings = {
    'Gender': {'Female': 0, 'Male': 1, 'Non-binary': 2},
    'Marital Status': {'Divorced': 0, 'Married': 1, 'Single': 2, 'Widowed': 3},
    'Loan Purpose': {'Auto': 0, 'Business': 1, 'Home': 2, 'Personal': 3},
    'Education Level': {"Bachelor's": 0, 'High School': 1, "Master's": 2, 'PhD': 3},
    'Employment Status': {'Employed': 0, 'Self-employed': 1, 'Unemployed': 2},
    'Payment History': {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Poor': 3}
}

# Risk levels mapping
risk_levels = {0: 'High', 1: 'Low', 2: 'Medium'}

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for the form page
@app.route('/form')
def form():
    # Pass input_columns if dynamic form fields are needed
    return render_template('form.html', input_columns=input_columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve inputs
    inputs = []
    for col in input_columns:
        val = request.form.get(col)
        if col in categorical_mappings:
            # Map categorical inputs to their numerical values
            mapped_val = categorical_mappings[col].get(val, None)
            if mapped_val is not None:
                inputs.append(mapped_val)
            else:
                return f"Invalid input for {col}: {val}", 400
        else:
            try:
                # Convert numeric inputs to float
                inputs.append(float(val))
            except ValueError:
                return f"Invalid numeric input for {col}: {val}", 400
    
    # Convert to array for prediction
    input_array = np.array(inputs).reshape(1, -1)
    
    # Predict using the model
    risk_prediction = model.predict(input_array)[0]  # Assuming the model outputs 0, 1, 2 for High, Low, Medium
    
    # Map prediction to risk levels
    risk = risk_levels.get(risk_prediction, "Unknown")
    
    # Generate suggestions (you can adjust this logic as per your needs)
    suggestions = {
        "Low": "Your risk level is low. Continue maintaining a healthy financial profile!",
        "Medium": "Your risk level is moderate. Consider improving your credit score and managing your debts better.",
        "High": "Your risk level is high. Focus on reducing your debts and improving your payment history."
    }
    
    # Return prediction and suggestion to the template
    return render_template('prediction.html', risk=risk, suggestion=suggestions.get(risk, "No suggestions available."))


if __name__ == '__main__':
    app.run(debug=True)

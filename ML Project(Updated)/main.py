from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
data = pd.read_csv('data.csv')  # Load the data.csv file

# Load the model
model_path = "real_estate_price_predictor.pkl"
try:
    pipe = joblib.load(model_path)
except Exception as e:
    print(f"Error loading the model: {e}")
    pipe = None

@app.route('/')
def index():
    # Extract unique sorted values for each feature
    crim = sorted(data['CRIM'].unique())
    zn = sorted(data['ZN'].unique())
    indus = sorted(data['INDUS'].unique())
    chas = sorted(data['CHAS'].unique())
    nox = sorted(data['NOX'].unique())
    rm = sorted(data['RM'].unique())
    age = sorted(data['AGE'].unique())
    dis = sorted(data['DIS'].unique())
    rad = sorted(data['RAD'].unique())
    tax = sorted(data['TAX'].unique())
    ptratio = sorted(data['PTRATIO'].unique())
    b = sorted(data['B'].unique())
    lstat = sorted(data['LSTAT'].unique())

    return render_template('index.html', crim=crim, zn=zn, indus=indus, chas=chas, nox=nox, rm=rm, age=age, dis=dis, rad=rad, tax=tax, ptratio=ptratio, b=b, lstat=lstat)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if pipe is None:
        return "Model not loaded properly."

    # Retrieve form data
    crim = float(request.form.get('crim'))
    zn = float(request.form.get('zn'))
    indus = float(request.form.get('indus'))
    chas = float(request.form.get('chas'))
    nox = float(request.form.get('nox'))
    rm = float(request.form.get('rm'))
    age = float(request.form.get('age'))
    dis = float(request.form.get('dis'))
    rad = float(request.form.get('rad'))
    tax = float(request.form.get('tax'))
    ptratio = float(request.form.get('ptratio'))
    b = float(request.form.get('b'))
    lstat = float(request.form.get('lstat'))

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]],
                              columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

    # Predict the price
    prediction = pipe.predict(input_data)[0]

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)

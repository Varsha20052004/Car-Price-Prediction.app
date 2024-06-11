from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess the dataset
def load_and_preprocess_data():
    car = pd.read_csv('quikr_car.csv')
    car = car[car['year'].str.isnumeric()]
    car['year'] = car['year'].astype(int)
    car = car[car['Price'] != 'Ask For Price']
    car['Price'] = car['Price'].str.replace(',', '').astype(int)
    car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
    car = car[car['kms_driven'].str.isnumeric()]
    car['kms_driven'] = car['kms_driven'].astype(int)
    car = car[~car['fuel_type'].isna()]
    car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
    car = car.reset_index(drop=True)
    return car

car = load_and_preprocess_data()

X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the pipeline
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                                       remainder='passthrough')

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)

pipe.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        company = request.form['company']
        year = int(request.form['year'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']

        input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], 
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        prediction = pipe.predict(input_data)[0]

        return render_template('result.html', prediction=round(prediction, 2))
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

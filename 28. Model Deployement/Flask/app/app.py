from flask import Flask
from flask import render_template, request
import joblib

model = joblib.load('loan_randomforest_model.pkl')
app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I Am in predict function")
    inp = [int(i) for i in request.form.values()]
    print(inp)
    result = model.predict([inp])
    print(result)
    if result[0]==1:
        res = "Sorry, Your Loan will be Rejected"
    else:
        res = "Congragulations, Your Loan will be Accepted"
    return render_template('index.html', prediction = res)


if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug = True)
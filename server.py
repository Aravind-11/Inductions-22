import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model_knn = pickle.load(open("iris-knn.pkl", 'rb'))
model_rf = pickle.load(open("iris-rf.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html", prediction_text = '')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    flower_data = [np.array(data[:4])]
    if(data[4]==1):
        model = model_knn
    elif(data[4]==2):
        model = model_rf
    prediction = model.predict(flower_data)
    if(prediction == 0):
        flower = "setosa"
    elif(prediction == 1):
        flower = "versicolor"
    else:
        flower = "virginica"

    return render_template('index.html', prediction_text = 'Species is {}'.format(flower))


if __name__ == '__main__':
    app.run(debug=True)

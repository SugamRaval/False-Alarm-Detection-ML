import pandas as pd
import numpy as np
from flask import Flask,request,render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))


@app.route("/predict", methods=['post'])
def predict():

    test = []
    test_input = request.get_json()
    for i in test_input.values():
        test.append(i)

    test = np.array(test).reshape(1,6)

    output = model.predict(test)

    # 1 - False Alarm
    # 0 - True Alarm

    if output[0] == 1:
        return "False Alarm, No Danger"
    else:
        return "True Alarm, Danger"


if __name__ == '__main__':
    app.run(port = 5001,debug=True)
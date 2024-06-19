import pandas as pd
import numpy as np
from flask import Flask,request,render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
result_df = pd.DataFrame(columns = ['Ambient Temp','Calibration','Unwanted Substance Deposition',
                                   'Humidity','H2S Content','Detected by(% of sensors)','Fire Alarm'])

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['post'])
def predict():
    global result_df

    input = [float(x) for x in request.form.values()]
    test = np.array(input).reshape(1,6)

    output = model.predict(test)
    input.append(output[0])

    result_df.loc[len(result_df)] = input
    print(result_df)
    result_df.to_csv('predicted_data.csv',index=False,header=False,mode='a')

    # 1 - False Alarm
    # 0 - True Alarm

    if output[0] == 1:
        return render_template('index.html',prediction_text="False Alarm, No Danger")
    else:
        return render_template('index.html',prediction_text="True Alarm, Danger")


if __name__ == '__main__':
    app.run(port = 5000,debug=True)
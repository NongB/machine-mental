from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('index.html',pred='คุณต้องได้รับการรักษา\nโอกาสเกิดอาการป่วยทางจิตอยู่ที่ {}'.format(output))
    else:
        return render_template('index.html',pred='คุณไม่จำเป็นต้องรักษา\n โอกาสป่วยทางจิตอยู่ที่ {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)

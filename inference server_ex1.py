from flask import Flask, request
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report
from loaded_model import *
from config import JSON_FILENAME

# import sklearn
# print(sklearn.__version__) # 0.23.2

app = Flask(__name__)


def take_test_data(json_file, y_test=False):
    """
    upload the test set.
    if y_test is in the test data (y_test=True) it will return the data with y_test,
    else just X_test
    :param json_file:
    :return: X_test and y_test
    """

    test = pd.read_json(json_file, orient='records')
    if y_test:
        return test.iloc[:, :-1], test.iloc[:, -1]
    else:
        return test


def prediction(the_model, x_test_data):
    """
    return the prediction of the X test data
    :param model:
    :param x_test_data:
    :return: y_pred
    """
    return the_model.predict(x_test_data)


def check_scores(y_test_data, y_pred_data):
    """
    give the model prediction scores if y i
    :param y_test_data:
    :param y_pred_data:
    :return: classification report
    """

    return classification_report(y_test_data, y_pred_data)


def url_create_dataframe(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                     free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol):
    """
    create X_train dataframe
    :rtype: all feature are numeric
    """
    X_new_data = np.array([[fixed_acidity], [volatile_acidity], [citric_acid], [residual_sugar], [chlorides],
                           [free_sulfur_dioxide], [total_sulfur_dioxide], [density], [ph], [sulphates], [alcohol]])
    x_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']
    return pd.DataFrame(data=X_new_data, index=x_columns).T


def create_pred_json(y_pred_data):
    y_pred_series = pd.Series(y_pred_data)
    return y_pred_series.to_json()


@app.route('/predict_single')
def predict_single():
    # X_pred = np.array([])
    # y_pred = np.array([])
    fixed_acidity = request.args.get('fixed_acidity')
    volatile_acidity = request.args.get('volatile_acidity')
    citric_acid = request.args.get('citric_acid')
    residual_sugar = request.args.get('residual_sugar')
    chlorides = request.args.get('chlorides')
    free_sulfur_dioxide = request.args.get('free_sulfur_dioxide')
    total_sulfur_dioxide = request.args.get('total_sulfur_dioxide')
    density = request.args.get('density')
    ph = request.args.get('ph')
    sulphates = request.args.get('sulphates')
    alcohol = request.args.get('alcohol')
    X_pred = url_create_dataframe(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                  free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol)
    y_pred = create_pred_json(prediction(model, X_pred))
    return y_pred


@app.route('/predict_multi', methods=['POST'])
def predict_multi():
    a = request.files['nofar'].read()
    X_test, y_test = take_test_data(a, y_test=True)
    y_pred = create_pred_json(prediction(model, X_test))
    return y_pred


if __name__ == "__main__":
    port = os.environ.get('PORT')

    if port:   # for heroku
        app.run(host='0.0.0.0', port=int(port))
    else:  # for local repo
        app.run(host='localhost',
                port=8000)  # 5000 was taken so I used 8000 - but it should work on your computer with 5000

# http://localhost:8000/predict_multi?fixed_acidity=12.3&volatile_acidity=9.8&citric_acid=0.49&residual_sugar=3.1&chlorides=0.08&free_sulfur_dioxide=28&total_sulfur_dioxide=46&density=0.9993&ph=3.2&sulphates=0.8&alcohol=10.2
# http://localhost:8000/predict_single?fixed_acidity=12.3&volatile_acidity=9.8&citric_acid=0.49&residual_sugar=3.1&chlorides=0.08&free_sulfur_dioxide=28&total_sulfur_dioxide=46&density=0.9993&ph=3.2&sulphates=0.8&alcohol=10.2

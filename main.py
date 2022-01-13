# MAIN PYTHON FILE

# importing neccessary packages & docs
import pickle
from types import MethodDescriptorType
from flask import Flask, request, jsonify
from model_files.ml_model import predict_mpg

# intialize flask app
app = Flask("mpg_prediction")

# HOME ROUTE
# @app.route('/', methods=['GET'])
# def ping():
#     return "Home Application"


@app.route('/', methods=['POST'])
def predict():
    vechile_config = request.get_json()

    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    predictions = predict_mpg(vechile_config, model)

    response = {
        'mpg_predictions': list(predictions)
    }

    return jsonify(response)



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9090)

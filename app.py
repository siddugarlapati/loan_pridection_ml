from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return "Loan Prediction Model is Running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    # The example features might differ from your model's actual features.
    # Please ensure they match what your model was trained on.
    features = np.array([[data["income"], data["loan_amount"], data["credit_history"]]])
    prediction = model.predict(features)
    return jsonify({"loan_approval": str(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

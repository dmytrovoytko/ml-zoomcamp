from flask import Flask, request, jsonify
import pickle


def load_models():
    with open('model1.bin', 'rb') as f_model:
        model = pickle.load(f_model)
    with open('dv.bin', 'rb') as f_dv:
        dv = pickle.load(f_dv)
    return model, dv

model, dv = load_models()
app = Flask('scoring')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    
    X = dv.transform([client])
    prediction = model.predict_proba(X)[0, 1]
    
    result = {
        'probability': float(prediction),
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

# gunicorn --bind 0.0.0.0:9696 predict:app 
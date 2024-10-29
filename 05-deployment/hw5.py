import requests
import pickle

def download_file(url, filename):
    try:
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}:", e)

def q3():
    prefix = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2024/05-deployment/homework"
    model_path='model1.bin'
    dv_path='dv.bin'
        
    try:
        download_file(f"{prefix}/{model_path}", model_path)
        download_file(f"{prefix}/{dv_path}", dv_path)

        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open(dv_path, 'rb') as dv_file:
            dv = pickle.load(dv_file)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return

    if model is not None and dv is not None:
        print("Model type:", type(model))
        print("DictVectorizer type:", type(dv))
        client_data = {"job": "management", "duration": 400, "poutcome": "success"}
        X = dv.transform([client_data])
        
        prediction = model.predict_proba(X)[0, 1]
        
        print(f"client_data: {client_data}\nprediction: {prediction:.4f}")

def q4():
    url = "http://localhost:9696/predict"
    client_data = {"job": "student", "duration": 280, "poutcome": "failure"}
    response = requests.post(url, json=client_data)
    print(response)
    print(response.json())

def q6():
    url = "http://localhost:9696/predict"
    client_data = {"job": "management", "duration": 400, "poutcome": "success"}
    response = requests.post(url, json=client_data)
    print(response)
    print(response.json())

if __name__ == "__main__":
    # q3()
    # q4()
    q6()
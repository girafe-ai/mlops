import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import uuid

MLFLOW_URL = "http://localhost:8888"

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    file.save(filepath)
    
    return jsonify({"path": filepath})

@app.route('/invocations', methods=['POST'])
def invocations():
    data = request.json
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(f"{MLFLOW_URL}/invocations", json=data, headers=headers)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8890)


from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import sys

app = Flask(__name__)
CORS(app)

MLFLOW_URL = os.getenv('MLFLOW_URL', 'http://mlflow:8888')

@app.route('/invocations', methods=['POST'])
def invocations():
    data = request.get_json()
    print(data, file=sys.stderr)
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(f"{MLFLOW_URL}/invocations", json=data, headers=headers)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        print(str(e), file=sys.stderr)
        # print(response.json(), file=sys.stderr)
        # print(jsonify(response.json()), file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    save_path = os.path.join('/uploads', file.filename)
    file.save(save_path)
    return jsonify({"path": save_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8890)

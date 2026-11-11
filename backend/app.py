from flask import Flask, jsonify
from flask_cors import CORS

from routes.analysis_routes import analysis_bp


app = Flask(__name__)


CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "https://ml-model-docter-1.onrender.com"
            ]
        }
    }
)

app.register_blueprint(analysis_bp)


@app.route("/")
def home():

    return jsonify({
        "message": "ML Model Doctor API is running!"
    })


@app.route("/health")
def health():

    return jsonify({
        "status": "healthy"
    })


if __name__ == "__main__":
    app.run(debug=True)

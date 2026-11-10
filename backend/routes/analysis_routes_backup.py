
from flask import Blueprint, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import os

from ml.data_loader import load_dataset, get_file_type
from ml.data_profiler import profile_dataset
from ml.data_cleaner import clean_dataset
from ml.model_engine import detect_problem_type, compare_models
from ml.recommendations import generate_recommendation


analysis_bp = Blueprint("analysis", __name__, url_prefix="/api")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CLEANED_FOLDER = os.path.join(UPLOAD_FOLDER, "cleaned")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".csv", ".xlsx", ".xls", ".json", ".parquet", ".zip"
}


@analysis_bp.route("/test", methods=["GET"])
def test_analysis():
    return jsonify({
        "status": "success",
        "message": "Analysis API is working!"
    })


@analysis_bp.route("/upload", methods=["POST"])
def upload_dataset():

    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded."
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "status": "error",
            "message": "No file selected."
        }), 400

    filename = secure_filename(file.filename)
    extension = os.path.splitext(filename)[1].lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "status": "error",
            "message": "Unsupported file format."
        }), 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(file_path)

        df = load_dataset(file_path)
        profile = profile_dataset(df)

        return jsonify({
            "status": "success",
            "file": {
                "filename": filename,
                "type": get_file_type(file_path)
            },
            "profile": profile,
            "message": (
                "Dataset loaded successfully. "
                "Select the target column to continue."
            )
        })

    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 400


@analysis_bp.route("/analyze", methods=["POST"])
def analyze_dataset():

    data = request.get_json(silent=True) or {}
    filename = secure_filename(data.get("filename", ""))
    target_column = data.get("target_column")

    if not filename or not target_column:
        return jsonify({
            "status": "error",
            "message": "Filename and target column are required."
        }), 400

    path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.isfile(path):
        return jsonify({
            "status": "error",
            "message": "Uploaded dataset not found. Upload again."
        }), 404

    try:
        original_df = load_dataset(path)

        if target_column not in original_df.columns:
            return jsonify({
                "status": "error",
                "message": "Selected target column does not exist."
            }), 400

        original_profile = profile_dataset(original_df)

        # Clean export and cleaning report
        cleaned_df, cleaning_report = clean_dataset(
            original_df,
            target_column=target_column
        )

        # Model evaluation uses the original features,
        # with target-missing rows excluded.
        # Feature preprocessing stays inside the train/test pipeline.
        model_df = original_df.dropna(
            subset=[target_column]
        ).copy()

        model_df = model_df.drop_duplicates().copy()

        X = model_df.drop(columns=[target_column])
        y = model_df[target_column]

        if X.shape[1] == 0 or y.nunique() < 2:
            return jsonify({
                "status": "error",
                "message": "Insufficient usable features or target classes."
            }), 400

        problem_type = detect_problem_type(y)

        profile = dict(original_profile)
        profile["original_missing_values"] = int(
            original_df.isna().sum().sum()
        )
        profile["original_duplicate_rows"] = int(
            original_df.duplicated().sum()
        )
        profile["target_missing_values"] = int(
            original_df[target_column].isna().sum()
        )
        profile["target_distribution"] = {
            str(k): int(v)
            for k, v in y.value_counts().to_dict().items()
        }

        comparison = compare_models(
            X,
            y,
            problem_type
        )

        recommendation = generate_recommendation(
            comparison,
            profile
        )

        # Save cleaned CSV for download.
        # A deterministic name is used for this prototype.
        cleaned_name = (
            os.path.splitext(filename)[0]
            + "_cleaned.csv"
        )
        cleaned_path = os.path.join(
            CLEANED_FOLDER,
            cleaned_name
        )
        cleaned_df.to_csv(cleaned_path, index=False)

        return jsonify({
            "status": "success",
            "dataset": {
                "filename": filename,
                "rows": int(model_df.shape[0]),
                "columns": int(model_df.shape[1])
            },
            "problem": {
                "type": problem_type,
                "target": target_column
            },
            "models": comparison["models"],
            "evaluation": {
                "method": comparison["evaluation_method"],
                "primary_metric": comparison["primary_metric"],
                "train_rows": comparison["train_rows"],
                "test_rows": comparison["test_rows"],
                "baseline": comparison["baseline"]
            },
            "recommendation": recommendation,
            "profile": profile,
            "cleaning_report": cleaning_report,
            "cleaned_file": cleaned_name
        })

    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 400


@analysis_bp.route("/download-cleaned/<path:filename>", methods=["GET"])
def download_cleaned(filename):

    safe_name = secure_filename(filename)

    if not safe_name.endswith("_cleaned.csv"):
        return jsonify({
            "status": "error",
            "message": "Invalid cleaned filename."
        }), 400

    path = os.path.join(CLEANED_FOLDER, safe_name)

    if not os.path.isfile(path):
        return jsonify({
            "status": "error",
            "message": "Cleaned dataset not found. Run analysis first."
        }), 404

    return send_from_directory(
        CLEANED_FOLDER,
        safe_name,
        as_attachment=True
    )

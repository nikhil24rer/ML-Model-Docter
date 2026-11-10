
import os
from uuid import uuid4

from flask import (
    Blueprint,
    jsonify,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from ml.data_loader import load_dataset, get_file_type
from ml.data_profiler import profile_dataset
from services.analysis_service import analyze_dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

analysis_bp = Blueprint(
    "analysis",
    __name__,
    url_prefix="/api",
)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CLEANED_FOLDER = os.path.join(UPLOAD_FOLDER, "cleaned")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet",
    ".zip",
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def error_response(message, status_code=400):
    """Return a consistent API error response."""

    return jsonify({
        "status": "error",
        "message": str(message),
    }), status_code


def get_safe_upload_path(filename):
    """Resolve a safe uploaded file path."""

    safe_filename = secure_filename(filename)

    if not safe_filename:
        raise ValueError("Invalid filename.")

    extension = os.path.splitext(safe_filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file format.")

    return safe_filename, os.path.join(
        UPLOAD_FOLDER,
        safe_filename,
    )


# ============================================================
# 3. API HEALTH TEST
# ============================================================

@analysis_bp.route("/test", methods=["GET"])
def test_analysis():

    return jsonify({
        "status": "success",
        "message": "Analysis API is working!",
    })


# ============================================================
# 4. UPLOAD DATASET
# ============================================================

@analysis_bp.route("/upload", methods=["POST"])
def upload_dataset():

    if "file" not in request.files:
        return error_response("No file uploaded.")

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return error_response("No file selected.")

    try:

        original_filename = secure_filename(
            uploaded_file.filename
        )

        extension = os.path.splitext(
            original_filename
        )[1].lower()

        if not original_filename or extension not in ALLOWED_EXTENSIONS:
            return error_response("Unsupported file format.")

        # Unique stored filename prevents accidental overwrite
        # when two users upload files with the same name.
        stored_filename = (
            f"{uuid4().hex}_{original_filename}"
        )

        file_path = os.path.join(
            UPLOAD_FOLDER,
            stored_filename,
        )

        uploaded_file.save(file_path)

        try:
            df = load_dataset(file_path)
            profile = profile_dataset(df)

        except Exception:
            # Do not keep invalid uploads.
            if os.path.isfile(file_path):
                os.remove(file_path)
            raise

        return jsonify({
            "status": "success",

            "file": {
                "filename": stored_filename,
                "original_filename": original_filename,
                "type": get_file_type(file_path),
            },

            "profile": profile,

            "message": (
                "Dataset loaded successfully. "
                "Select the target column to continue."
            ),
        })

    except ValueError as exc:
        return error_response(exc, 400)

    except Exception:
        # Avoid exposing internal exception details to clients.
        return error_response(
            "Unable to process the uploaded dataset.",
            500,
        )


# ============================================================
# 5. ANALYZE DATASET
# ============================================================

@analysis_bp.route("/analyze", methods=["POST"])
def analyze_uploaded_dataset():

    data = request.get_json(silent=True) or {}

    filename = data.get("filename")
    target_column = data.get("target_column")

    if not filename or not target_column:
        return error_response(
            "Filename and target column are required."
        )

    try:

        filename, file_path = get_safe_upload_path(filename)

    except ValueError as exc:
        return error_response(exc, 400)

    if not os.path.isfile(file_path):
        return error_response(
            "Uploaded dataset not found. Upload again.",
            404,
        )

    try:

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        df = load_dataset(file_path)

        if target_column not in df.columns:
            return error_response(
                "Selected target column does not exist."
            )

        # ----------------------------------------------------
        # Read optional analysis settings
        # ----------------------------------------------------

        user_choice = data.get("problem_type")

        max_models = data.get("max_models", 6)
        test_size = data.get("test_size", 0.20)

        if isinstance(max_models, bool) or not isinstance(max_models, int):
            return error_response(
                "max_models must be an integer."
            )

        if isinstance(test_size, bool) or not isinstance(
            test_size, (int, float)
        ):
            return error_response(
                "test_size must be a number."
            )

        # ----------------------------------------------------
        # Run central analysis service
        # ----------------------------------------------------

        result = analyze_dataset(
            df=df,
            target_column=target_column,
            user_choice=user_choice,
            max_models=max_models,
            test_size=float(test_size),
            generate_reports=True,
            report_directory=REPORT_FOLDER,
        )

        # Ambiguous targets need confirmation from the user.
        if result["status"] == "target_confirmation_required":

            return jsonify(result), 200

        if result["status"] != "success":
            return error_response(
                result.get("message", "Analysis could not be completed.")
            )

        # ----------------------------------------------------
        # Save cleaned dataset
        # ----------------------------------------------------

        cleaned_df = result["cleaned_dataframe"]

        cleaned_name = (
            f"{os.path.splitext(filename)[0]}_cleaned.csv"
        )

        cleaned_path = os.path.join(
            CLEANED_FOLDER,
            cleaned_name,
        )

        cleaned_df.to_csv(
            cleaned_path,
            index=False,
        )

        # ----------------------------------------------------
        # Prepare public API response
        # ----------------------------------------------------

        report_paths = result.get("report_paths", {})

        json_report_name = (
            os.path.basename(report_paths["json"])
            if report_paths.get("json")
            else None
        )

        html_report_name = (
            os.path.basename(report_paths["html"])
            if report_paths.get("html")
            else None
        )

        response = {
            "status": "success",

            "dataset": {
                "filename": filename,
                **result["dataset"],
            },

            "problem": result["problem"],

            # Preserve existing frontend response key.
            "models": result["models"],

            "model_comparison": result["model_comparison"],

            "evaluation": result["evaluation"],

            "recommendation": result["recommendation"],

            "diagnosis": result["diagnosis"],

            "profile": result["profile"],

            "cleaning_report": result["cleaning_report"],

            "cleaned_file": cleaned_name,

            "reports": {
                "json": json_report_name,
                "html": html_report_name,
            },

            "execution": result["execution"],

            "message": "Dataset analysis completed successfully.",
        }

        return jsonify(response), 200

    except ValueError as exc:
        return error_response(exc, 400)

    except Exception:
        # Log the actual exception in the server terminal.
        import traceback
        traceback.print_exc()

        return error_response(
            "An unexpected error occurred during dataset analysis.",
            500,
        )


# ============================================================
# 6. DOWNLOAD CLEANED DATASET
# ============================================================

@analysis_bp.route(
    "/download-cleaned/<path:filename>",
    methods=["GET"],
)
def download_cleaned(filename):

    safe_name = secure_filename(filename)

    if not safe_name.endswith("_cleaned.csv"):
        return error_response("Invalid cleaned filename.")

    path = os.path.join(
        CLEANED_FOLDER,
        safe_name,
    )

    if not os.path.isfile(path):
        return error_response(
            "Cleaned dataset not found. Run analysis first.",
            404,
        )

    return send_from_directory(
        CLEANED_FOLDER,
        safe_name,
        as_attachment=True,
    )


# ============================================================
# 7. DOWNLOAD JSON REPORT
# ============================================================

@analysis_bp.route(
    "/download-report/json/<path:filename>",
    methods=["GET"],
)
def download_json_report(filename):

    safe_name = secure_filename(filename)

    if not safe_name.endswith(".json"):
        return error_response("Invalid JSON report filename.")

    path = os.path.join(
        REPORT_FOLDER,
        safe_name,
    )

    if not os.path.isfile(path):
        return error_response("JSON report not found.", 404)

    return send_from_directory(
        REPORT_FOLDER,
        safe_name,
        as_attachment=True,
    )


# ============================================================
# 8. DOWNLOAD HTML REPORT
# ============================================================

@analysis_bp.route(
    "/download-report/html/<path:filename>",
    methods=["GET"],
)
def download_html_report(filename):

    safe_name = secure_filename(filename)

    if not safe_name.endswith(".html"):
        return error_response("Invalid HTML report filename.")

    path = os.path.join(
        REPORT_FOLDER,
        safe_name,
    )

    if not os.path.isfile(path):
        return error_response("HTML report not found.", 404)

    return send_from_directory(
        REPORT_FOLDER,
        safe_name,
        as_attachment=True,
    )


import json
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================
# 1. SERIALIZATION HELPERS
# ============================================================

def make_json_safe(value):
    """Convert common Python and NumPy values into JSON-safe values."""

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

    except ImportError:
        pass

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


# ============================================================
# 2. REPORT VALIDATION
# ============================================================

def validate_report_inputs(
    dataset_profile=None,
    target_info=None,
    model_results=None,
    evaluation_report=None,
    diagnosis_report=None,
    explainability_report=None,
    tuning_report=None,
):
    """Ensure all supplied report sections are dictionaries."""

    sections = {
        "dataset_profile": dataset_profile,
        "target_info": target_info,
        "model_results": model_results,
        "evaluation_report": evaluation_report,
        "diagnosis_report": diagnosis_report,
        "explainability_report": explainability_report,
        "tuning_report": tuning_report,
    }

    for name, section in sections.items():

        if section is not None and not isinstance(section, dict):
            raise ValueError(
                f"{name} must be a dictionary or None."
            )

    return sections


# ============================================================
# 3. EXTRACT BEST MODEL
# ============================================================

def extract_best_model(model_results):
    """Extract best model information from comparison results."""

    if not isinstance(model_results, dict):
        return None

    best_model = model_results.get("best_model")

    if best_model is None:
        best_model = model_results.get("best_model_name")

    if isinstance(best_model, dict):
        return make_json_safe(best_model)

    if best_model is not None:
        return {"name": str(best_model)}

    return None


# ============================================================
# 4. BUILD EXECUTIVE SUMMARY
# ============================================================

def build_executive_summary(
    dataset_profile=None,
    target_info=None,
    model_results=None,
    evaluation_report=None,
    diagnosis_report=None,
):
    """Create a short human-readable summary."""

    dataset_profile = dataset_profile or {}
    target_info = target_info or {}
    model_results = model_results or {}
    evaluation_report = evaluation_report or {}
    diagnosis_report = diagnosis_report or {}

    best_model = extract_best_model(model_results)

    return {
        "dataset": {
            "rows": dataset_profile.get("rows"),
            "columns": dataset_profile.get("columns"),
        },
        "problem_type": target_info.get("problem_type"),
        "best_model": best_model,
        "evaluation": {
            "generalization": evaluation_report.get("generalization"),
            "baseline_comparison": evaluation_report.get(
                "baseline_comparison"
            ),
        },
        "diagnosis_status": diagnosis_report.get("status"),
        "diagnostic_findings": (
            diagnosis_report.get("summary", {})
        ),
        "conclusion": (
            "Review the evaluation metrics and diagnostic findings "
            "before using the model in production."
        ),
    }


# ============================================================
# 5. BUILD COMPLETE REPORT
# ============================================================

def generate_report(
    dataset_profile=None,
    target_info=None,
    model_results=None,
    evaluation_report=None,
    diagnosis_report=None,
    explainability_report=None,
    tuning_report=None,
    metadata=None,
):
    """
    Combine outputs from the ML Model Doctor modules.

    Missing sections are represented as None rather than invented.
    """

    sections = validate_report_inputs(
        dataset_profile=dataset_profile,
        target_info=target_info,
        model_results=model_results,
        evaluation_report=evaluation_report,
        diagnosis_report=diagnosis_report,
        explainability_report=explainability_report,
        tuning_report=tuning_report,
    )

    metadata = metadata or {}

    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary.")

    report = {
        "report_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "metadata": metadata,

        "executive_summary": build_executive_summary(
            sections["dataset_profile"],
            sections["target_info"],
            sections["model_results"],
            sections["evaluation_report"],
            sections["diagnosis_report"],
        ),

        "dataset_profile": sections["dataset_profile"],
        "target_analysis": sections["target_info"],
        "model_comparison": sections["model_results"],
        "evaluation": sections["evaluation_report"],
        "diagnosis": sections["diagnosis_report"],
        "explainability": sections["explainability_report"],
        "hyperparameter_tuning": sections["tuning_report"],
    }

    return make_json_safe(report)


# ============================================================
# 6. JSON REPORT EXPORT
# ============================================================

def export_json_report(
    report,
    output_path="reports/ml_model_report.json",
):
    """Export a report to a JSON file."""

    if not isinstance(report, dict):
        raise ValueError("Report must be a dictionary.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_report = make_json_safe(report)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            safe_report,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    return str(path)


# ============================================================
# 7. HTML REPORT GENERATION
# ============================================================

def format_html_value(value):
    """Format values safely for HTML."""

    value = make_json_safe(value)

    if value is None:
        return "Not available"

    if isinstance(value, (dict, list)):
        return (
            "<pre>"
            + html.escape(
                json.dumps(value, indent=2, ensure_ascii=False)
            )
            + "</pre>"
        )

    return html.escape(str(value))


def build_html_report(report):
    """Build a standalone HTML report."""

    if not isinstance(report, dict):
        raise ValueError("Report must be a dictionary.")

    safe_report = make_json_safe(report)

    sections = [
        ("Executive Summary", safe_report.get("executive_summary")),
        ("Dataset Profile", safe_report.get("dataset_profile")),
        ("Target Analysis", safe_report.get("target_analysis")),
        ("Model Comparison", safe_report.get("model_comparison")),
        ("Evaluation", safe_report.get("evaluation")),
        ("Diagnosis", safe_report.get("diagnosis")),
        ("Explainability", safe_report.get("explainability")),
        (
            "Hyperparameter Tuning",
            safe_report.get("hyperparameter_tuning"),
        ),
    ]

    cards = []

    for title, content in sections:

        cards.append(
            f"""
            <section class="card">
                <h2>{html.escape(title)}</h2>
                {format_html_value(content)}
            </section>
            """
        )

    generated_at = html.escape(
        str(safe_report.get("generated_at", "Not available"))
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Model Doctor Report</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 30px;
    font-family: Arial, sans-serif;
    background: #f4f6fa;
    color: #202938;
}}

.container {{
    max-width: 1100px;
    margin: auto;
}}

header {{
    background: #182338;
    color: white;
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
}}

header h1 {{
    margin: 0 0 10px;
}}

.card {{
    background: white;
    padding: 22px;
    margin-bottom: 18px;
    border-radius: 10px;
    border: 1px solid #e1e6ed;
    overflow-wrap: anywhere;
}}

.card h2 {{
    margin-top: 0;
    color: #244a80;
}}

pre {{
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    background: #f4f6fa;
    padding: 14px;
    border-radius: 6px;
}}

footer {{
    text-align: center;
    color: #667085;
    margin-top: 25px;
}}

@media print {{
    body {{
        background: white;
        padding: 10px;
    }}

    .card, header {{
        break-inside: avoid;
    }}
}}
</style>
</head>

<body>
<div class="container">

<header>
    <h1>ML Model Doctor</h1>
    <p>Automated Machine Learning Analysis Report</p>
    <small>Generated: {generated_at}</small>
</header>

{''.join(cards)}

<footer>
    Generated by ML Model Doctor
</footer>

</div>
</body>
</html>"""


# ============================================================
# 8. HTML REPORT EXPORT
# ============================================================

def export_html_report(
    report,
    output_path="reports/ml_model_report.html",
):
    """Save the HTML report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    html_content = build_html_report(report)

    with path.open("w", encoding="utf-8") as file:
        file.write(html_content)

    return str(path)


# ============================================================
# 9. DASHBOARD SUMMARY
# ============================================================

def build_dashboard_summary(report):
    """Return a compact summary suitable for frontend rendering."""

    if not isinstance(report, dict):
        raise ValueError("Report must be a dictionary.")

    summary = report.get("executive_summary", {})

    return make_json_safe({
        "generated_at": report.get("generated_at"),
        "dataset": summary.get("dataset"),
        "problem_type": summary.get("problem_type"),
        "best_model": summary.get("best_model"),
        "diagnosis_status": summary.get("diagnosis_status"),
        "diagnostic_findings": summary.get("diagnostic_findings"),
        "generalization": (
            summary.get("evaluation", {}).get("generalization")
        ),
    })


# ============================================================
# 10. SAVE COMPLETE REPORT
# ============================================================

def save_report(
    report,
    json_path="reports/ml_model_report.json",
    html_path="reports/ml_model_report.html",
):
    """Export both JSON and HTML reports."""

    json_file = export_json_report(report, json_path)
    html_file = export_html_report(report, html_path)

    return {
        "json_path": json_file,
        "html_path": html_file,
    }

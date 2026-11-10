import { useState } from "react";
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  Database,
  Download,
  FileSpreadsheet,
  FileText,
  HeartPulse,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  UploadCloud,
  AlertTriangle,
  ChartNoAxesColumn,
  Stethoscope,
} from "lucide-react";

import {
  uploadDataset,
  analyzeDataset,
  getDownloadUrl,
} from "./services/api";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

import "./App.css";

const PROBLEM_TYPES = [
  { value: "binary_classification", label: "Binary Classification" },
  { value: "multiclass_classification", label: "Multiclass Classification" },
  { value: "regression", label: "Regression" },
];

const formatNumber = (value, digits = 3) => {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(digits);
};

const formatLabel = (value) =>
  String(value ?? "—")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

function MetricCard({ icon: Icon, label, value, detail }) {
  return (
    <div className="metric-card">
      <div className="metric-icon"><Icon size={19} /></div>
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function SectionHeading({ eyebrow, title, description }) {
  return (
    <div className="section-heading">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
    </div>
  );
}

function ScoreRow({ label, value }) {
  return (
    <div className="score-row">
      <span>{label}</span>
      <strong>{formatNumber(value)}</strong>
    </div>
  );
}

function DownloadButton({ label, type, filename, icon: Icon }) {
  if (!filename) return null;

  function download() {
    const url = getDownloadUrl(type, filename);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  return (
    <button className="download-button" onClick={download}>
      <Icon size={17} />
      {label}
    </button>
  );
}


function ConfusionMatrix({ matrix, labels = [] }) {
  const validMatrix =
    Array.isArray(matrix) &&
    matrix.length > 0 &&
    matrix.every(
      (row) =>
        Array.isArray(row) &&
        row.length === matrix.length &&
        row.every((value) => Number.isFinite(Number(value)))
    );

  if (!validMatrix) {
    return <p className="matrix-empty">Confusion matrix is not available.</p>;
  }

  const classLabels = matrix.map((_, index) =>
    labels[index] ?? `Class ${index + 1}`
  );

  return (
    <div className="confusion-matrix-wrap">
      <table className="confusion-matrix">
        <thead>
          <tr>
            <th rowSpan={2}>Actual ↓ / Predicted →</th>
            <th colSpan={matrix.length}>Predicted class</th>
          </tr>
          <tr>
            {classLabels.map((label) => (
              <th key={label}>{label}</th>
            ))}
          </tr>
        </thead>

        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <th>{classLabels[rowIndex]}</th>
              {row.map((value, colIndex) => (
                <td
                  key={colIndex}
                  className={
                    rowIndex === colIndex ? "matrix-correct" : "matrix-incorrect"
                  }
                >
                  {Number(value).toLocaleString()}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <div className="matrix-legend">
        <span><i className="matrix-correct-dot" /> Correct predictions</span>
        <span><i className="matrix-incorrect-dot" /> Incorrect predictions</span>
      </div>
    </div>
  );
}


function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [target, setTarget] = useState("");
  const [problemType, setProblemType] = useState("");
  const [maxModels, setMaxModels] = useState(6);
  const [testSize, setTestSize] = useState(0.2);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const profile = upload?.profile;
  const evaluation = result?.evaluation;
  const evaluationDetails = evaluation?.details;
  const recommendation = result?.recommendation;
  const diagnosis = result?.diagnosis;
  const cleaning = result?.cleaning_report;

  const isClassification = [
    "classification",
    "binary_classification",
    "multiclass_classification",
  ].includes(result?.problem?.type);

  function resetMessages() {
    setError("");
    setSuccess("");
  }

  function chooseFile(selectedFile) {
    if (!selectedFile) return;

    resetMessages();
    setResult(null);
    setUpload(null);
    setTarget("");
    setProblemType("");

    const allowed = [".csv", ".xlsx", ".xls", ".json", ".parquet", ".zip"];
    const valid = allowed.some((ext) =>
      selectedFile.name.toLowerCase().endsWith(ext)
    );

    if (!valid) {
      setFile(null);
      setError("Unsupported file. Please upload CSV, Excel, JSON, Parquet or ZIP.");
      return;
    }

    setFile(selectedFile);
  }

  async function handleUpload() {
    if (!file) {
      setError("Please select a dataset first.");
      return;
    }

    setLoading(true);
    resetMessages();

    try {
      const data = await uploadDataset(file);

      setUpload(data);
      setSuccess("Dataset uploaded and profiled successfully.");

      const columns = data.profile?.column_names || [];
      if (columns.length) {
        setTarget(columns[columns.length - 1]);
      }
    } catch (err) {
      setError(err.message || "Dataset upload failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleAnalysis() {
    if (!upload || !target || !problemType) {
      setError("Please select the target column and confirm the problem type.");
      return;
    }

    setLoading(true);
    setResult(null);
    resetMessages();

    try {
      const data = await analyzeDataset({
        filename: upload.file.filename,
        targetColumn: target,
        problemType,
        maxModels,
        testSize,
      });

      if (data.status === "target_confirmation_required") {
        setError(
          data.message ||
          "The target needs confirmation. Please select the correct problem type."
        );
        return;
      }
      setResult(data);
setSuccess("Model analysis completed successfully.");
    } catch (err) {
      setError(err.message || "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function resetAll() {
    setFile(null);
    setUpload(null);
    setTarget("");
    setProblemType("");
    setResult(null);
    resetMessages();
  }

  const successfulModels = (result?.models || []).filter(
    (model) => model.status === "success" && !model.error
  );

  const chartData = successfulModels.map((model) => ({
    name: model.model,
    cvScore: Number(model.cv_mean_score ?? 0),
    testScore: Number(
      isClassification ? model.f1 ?? 0 : model.r2 ?? 0
    ),
  }));

  // The backend exposes explainability at the top level and also inside
  // model_comparison. Prefer the top-level report, then use the nested one.
  const explainability =
    result?.explainability ??
    result?.model_comparison?.explainability ??
    {};

  const importanceCandidates = [
    {
      method: "Native feature importance",
      data: explainability?.native_importance,
    },
    {
      method: "Model coefficients",
      data: explainability?.coefficients,
    },
    {
      method: "Permutation importance",
      data: explainability?.permutation_importance,
    },
  ];

  // Do not stop at an available method if it has no usable feature records.
  const importanceData =
    importanceCandidates.find(({ data }) => {
      const records = data?.all_features ?? data?.top_features;
      return data?.available === true &&
        Array.isArray(records) &&
        records.length > 0;
    }) ?? null;

  const importanceItems = importanceData
    ? (
        importanceData.data.all_features ??
        importanceData.data.top_features ??
        []
      )
    : [];

  const importanceMethod =
    importanceData?.data?.method ??
    importanceData?.method ??
    "Feature importance";

  const normalizedImportance = importanceItems
    .map((item) => ({
      feature: String(
        item?.feature ??
        item?.name ??
        item?.feature_name ??
        ""
      ),
      importance: Number(
        item?.importance ??
        item?.score ??
        item?.value
      ),
    }))
    .filter(
      (item) =>
        item.feature.trim().length > 0 &&
        Number.isFinite(item.importance)
    )
    .sort(
      (a, b) =>
        Math.abs(b.importance) - Math.abs(a.importance)
    )
    .slice(0, 10);

  // sklearn's confusion_matrix uses sorted class labels by default.
  // Numeric target values are sorted numerically; string labels alphabetically.
  const targetLabels = Object.keys(
    result?.profile?.target_distribution ?? {}
  ).sort((a, b) => {
    const numberA = Number(a);
    const numberB = Number(b);
    const bothNumeric =
      a.trim() !== "" &&
      b.trim() !== "" &&
      Number.isFinite(numberA) &&
      Number.isFinite(numberB);

    return bothNumeric
      ? numberA - numberB
      : a.localeCompare(b);
  });

  const findings = diagnosis?.findings || [];
  const suggestions = recommendation?.suggestions || [];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-logo"><Stethoscope size={23} /></div>
          <div>
            <strong>Model Doctor</strong>
            <span>ML DIAGNOSTICS</span>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <a href="#overview" className="sidebar-link active">
          <Activity size={18} /> Overview
        </a>
        <a href="#dataset" className="sidebar-link">
          <Database size={18} /> Dataset
        </a>
        <a href="#models" className="sidebar-link">
          <BrainCircuit size={18} /> Model Analysis
        </a>
        <a href="#diagnosis" className="sidebar-link">
          <HeartPulse size={18} /> Diagnosis
        </a>
        <a href="#reports" className="sidebar-link">
          <FileText size={18} /> Reports
        </a>

        <div className="sidebar-bottom">
          <div className="backend-status">
            <span className="online-dot" />
            Backend API
            <strong>Connected*</strong>
          </div>
          <small>*Connection is verified when an API request succeeds.</small>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div>
            <span className="topbar-label">MACHINE LEARNING WORKSPACE</span>
            <h1>ML Model Doctor</h1>
          </div>
          <button className="reset-button" onClick={resetAll}>
            <RefreshCw size={16} /> New Analysis
          </button>
        </header>

        <section className="welcome-section">
          <div>
            <span className="welcome-badge">
              <ShieldCheck size={15} /> Intelligent ML Diagnostics
            </span>
            <h2>Understand your data.<br /><span>Build better models.</span></h2>
            <p>
              Analyze dataset quality, compare machine learning models,
              identify potential issues and generate diagnostic reports.
            </p>
          </div>
          <div className="welcome-art">
            <div className="art-circle"><BrainCircuit size={72} /></div>
            <span className="art-dot dot-one" />
            <span className="art-dot dot-two" />
            <span className="art-dot dot-three" />
          </div>
        </section>

        {error && (
          <div className="alert alert-error">
            <AlertTriangle size={19} />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="alert alert-success">
            <CheckCircle2 size={19} />
            <span>{success}</span>
          </div>
        )}

        <section className="panel upload-panel" id="dataset">
          <SectionHeading
            eyebrow="STEP 01 · DATA INPUT"
            title="Upload your dataset"
            description="Start by uploading the tabular dataset you want to inspect."
          />

          <label className="drop-zone" htmlFor="dataset-file">
            <div className="upload-symbol"><UploadCloud size={30} /></div>
            <strong>Choose a dataset from your computer</strong>
            <span>CSV, XLSX, XLS, JSON, Parquet or ZIP</span>
            <span className="browse-text">Browse files</span>
            <input
              id="dataset-file"
              type="file"
              accept=".csv,.xlsx,.xls,.json,.parquet,.zip"
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />
          </label>

          {file && (
            <div className="selected-file">
              <div className="file-symbol"><FileSpreadsheet size={21} /></div>
              <div className="file-details">
                <strong>{file.name}</strong>
                <span>{(file.size / 1024).toFixed(1)} KB</span>
              </div>
              <button
                className="remove-file"
                onClick={() => {
                  setFile(null);
                  setUpload(null);
                  setResult(null);
                  setTarget("");
                  setProblemType("");
                }}
              >
                Remove
              </button>
            </div>
          )}

          {file && !upload && (
            <button
              className="primary-button"
              onClick={handleUpload}
              disabled={loading}
            >
              {loading ? <LoaderCircle className="spin" size={18} /> : <UploadCloud size={18} />}
              {loading ? "Uploading..." : "Upload & Profile Dataset"}
            </button>
          )}
        </section>

        {profile && (
          <section className="panel">
            <SectionHeading
              eyebrow="DATASET OVERVIEW"
              title="Dataset health"
              description={`Uploaded file: ${upload.file.original_filename || file?.name}`}
            />

            <div className="metrics-grid">
              <MetricCard icon={Database} label="Total Rows" value={profile.rows} />
              <MetricCard icon={ChartNoAxesColumn} label="Total Columns" value={profile.columns} />
              <MetricCard icon={AlertTriangle} label="Missing Values" value={profile.missing_values} />
              <MetricCard icon={RefreshCw} label="Duplicate Rows" value={profile.duplicate_rows} />
            </div>

            <div className="profile-details">
              <div>
                <span>Numeric columns</span>
                <strong>{profile.numeric_columns?.length ?? 0}</strong>
              </div>
              <div>
                <span>Categorical columns</span>
                <strong>{profile.categorical_columns?.length ?? 0}</strong>
              </div>
              <div>
                <span>Constant columns</span>
                <strong>{profile.constant_columns?.length ?? 0}</strong>
              </div>
              <div>
                <span>Identifier candidates</span>
                <strong>{profile.identifier_candidates?.length ?? 0}</strong>
              </div>
            </div>

            <div className="configuration-box">
              <SectionHeading
                eyebrow="STEP 02 · ANALYSIS SETTINGS"
                title="Configure your analysis"
                description="Choose the outcome to predict and confirm the ML problem type."
              />

              <div className="form-grid">
                <label className="form-field">
                  <span>Target column</span>
                  <select value={target} onChange={(e) => setTarget(e.target.value)}>
                    <option value="">Select target</option>
                    {(profile.column_names || []).map((column) => (
                      <option value={column} key={column}>{column}</option>
                    ))}
                  </select>
                </label>

                <label className="form-field">
                  <span>Problem type</span>
                  <select
                    value={problemType}
                    onChange={(e) => setProblemType(e.target.value)}
                  >
                    <option value="">Confirm problem type</option>
                    {PROBLEM_TYPES.map((item) => (
                      <option value={item.value} key={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="form-field">
                  <span>Maximum models</span>
                  <select
                    value={maxModels}
                    onChange={(e) => setMaxModels(Number(e.target.value))}
                  >
                    {[3, 4, 5, 6, 8].map((n) => (
                      <option key={n} value={n}>{n} models</option>
                    ))}
                  </select>
                </label>

                <label className="form-field">
                  <span>Test set size</span>
                  <select
                    value={testSize}
                    onChange={(e) => setTestSize(Number(e.target.value))}
                  >
                    <option value={0.2}>20%</option>
                    <option value={0.25}>25%</option>
                    <option value={0.3}>30%</option>
                  </select>
                </label>
              </div>

              <button
                className="primary-button"
                onClick={handleAnalysis}
                disabled={loading || !target || !problemType}
              >
                {loading ? <LoaderCircle className="spin" size={18} /> : <BrainCircuit size={18} />}
                {loading ? "Training and evaluating models..." : "Start Model Diagnosis"}
              </button>
            </div>
          </section>
        )}

        {result && (
          <>
            <section className="panel result-panel">
              <div className="result-heading">
                <div>
                  <span className="eyebrow">ANALYSIS COMPLETE</span>
                  <h2>Your model diagnostic report</h2>
                  <p>{result.message}</p>
                </div>
                <span className="complete-badge"><CheckCircle2 size={16} /> Completed</span>
              </div>

              <div className="metrics-grid">
                <MetricCard icon={BrainCircuit} label="Problem Type" value={formatLabel(result.problem?.type)} />
                <MetricCard icon={Database} label="Training Rows" value={evaluation?.train_rows} />
                <MetricCard icon={ChartNoAxesColumn} label="Testing Rows" value={evaluation?.test_rows} />
                <MetricCard icon={Activity} label="CV Folds" value={evaluation?.cv_folds} />
              </div>

              <div className="recommended-box">
                <div className="recommend-icon"><ShieldCheck size={27} /></div>
                <div>
                  <span className="eyebrow">RECOMMENDED MODEL</span>
                  <h2>{recommendation?.recommended_model || "No recommendation"}</h2>
                  <p>{recommendation?.reason || "No recommendation details available."}</p>
                </div>
              </div>

              <div className="score-summary">
                <div>
                  <span>Best CV score</span>
                  <strong>{formatNumber(evaluation?.best_cv_score)}</strong>
                </div>
                <div>
                  <span>Holdout test score</span>
                  <strong>{formatNumber(evaluation?.best_test_score)}</strong>
                </div>
                <div>
                  <span>Baseline F1 / R²</span>
                  <strong>
                    {formatNumber(
                      isClassification
                        ? evaluation?.baseline?.f1
                        : evaluation?.baseline?.r2
                    )}
                  </strong>
                </div>
                <div>
                  <span>CV standard deviation</span>
                  <strong>{formatNumber(evaluation?.best_cv_std)}</strong>
                </div>
              </div>
              <p className="method-note">
                {evaluation?.method}. {evaluation?.score_interpretation}
              </p>
            </section>

            <section className="panel" id="models">
              <SectionHeading
                eyebrow="MODEL BENCHMARKING"
                title="Model comparison"
                description="Compare cross-validation performance with final holdout performance."
              />

              <div className="chart-container">
                <ResponsiveContainer width="100%" height={310}>
                  <BarChart data={chartData} margin={{ top: 15, right: 15, left: 0, bottom: 55 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="name"
                      angle={-20}
                      textAnchor="end"
                      interval={0}
                      height={75}
                      tick={{ fontSize: 11 }}
                    />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Bar dataKey="cvScore" name="CV Score" fill="#7568e8" radius={[5, 5, 0, 0]} />
                    <Bar dataKey="testScore" name="Test Score" fill="#20b99a" radius={[5, 5, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="chart-legend">
                  <span><i className="legend-purple" /> Cross-validation</span>
                  <span><i className="legend-green" /> Holdout test</span>
                </div>
              </div>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>CV Mean</th>
                      <th>CV Std</th>
                      <th>{isClassification ? "Test F1" : "Test R²"}</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(result.models || []).map((model) => (
                      <tr key={model.model_id || model.model}>
                        <td><strong>{model.model}</strong></td>
                        <td>{formatNumber(model.cv_mean_score)}</td>
                        <td>{formatNumber(model.cv_std_score)}</td>
                        <td>
                          {formatNumber(
                            isClassification ? model.f1 : model.r2
                          )}
                        </td>
                        <td>
                          {model.status === "success" ? (
                            <span className="status-success">Successful</span>
                          ) : (
                            <span className="status-failed">Failed</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="panel">
              <SectionHeading
                eyebrow="MODEL PERFORMANCE"
                title="Detailed evaluation"
                description="Metrics calculated on the training and held-out test datasets."
              />

              <div className="model-detail-grid">
                {successfulModels.map((model) => (
                  <div className="model-detail-card" key={model.model_id || model.model}>
                    <h3>{model.model}</h3>
                    {isClassification ? (
                      <>
                        <ScoreRow label="Accuracy" value={model.accuracy} />
                        <ScoreRow label="Balanced accuracy" value={model.balanced_accuracy} />
                        <ScoreRow label="Precision" value={model.precision} />
                        <ScoreRow label="Recall" value={model.recall} />
                        <ScoreRow label="Macro F1" value={model.f1} />
                        
                    <details className="matrix-details">
                      <summary>View confusion matrix</summary>
                      <ConfusionMatrix
                        matrix={model.confusion_matrix}
                        labels={targetLabels}
                      />
                    </details>

                      </>
                    ) : (
                      <>
                        <ScoreRow label="R²" value={model.r2} />
                        <ScoreRow label="MAE" value={model.mae} />
                        <ScoreRow label="RMSE" value={model.rmse} />
                      </>
                    )}
                  </div>
                ))}
              </div>
            </section>

            <section className="panel" id="diagnosis">
              <SectionHeading
                eyebrow="MODEL HEALTH"
                title="Diagnosis and recommendations"
                description="Heuristic observations to help investigate possible model and dataset issues."
              />

              <div className="diagnosis-list">
                {findings.length ? findings.map((finding, index) => (
                  <div className="finding-card" key={finding.code || index}>
                    <div className={`finding-indicator ${finding.severity || "info"}`} />
                    <div>
                      <div className="finding-title">
                        <h3>{finding.title}</h3>
                        <span>{formatLabel(finding.severity)}</span>
                      </div>
                      <p>{finding.message}</p>
                      {finding.recommendation && (
                        <small>Suggested action: {finding.recommendation}</small>
                      )}
                    </div>
                  </div>
                )) : (
                  <p>No diagnostic findings were returned.</p>
                )}
              </div>

              <div className="suggestions-box">
                <h3><HeartPulse size={19} /> Recommended actions</h3>
                {suggestions.length ? (
                  <ul>{suggestions.map((item, index) => <li key={index}>{item}</li>)}</ul>
                ) : <p>No additional suggestions available.</p>}
              </div>

              {evaluationDetails && (
                <div className="evaluation-checks">
                  <h3>Evaluation checks</h3>
                  <div className="check-grid">
                    <div>
                      <span>Generalization</span>
                      <strong>{formatLabel(evaluationDetails.generalization?.status)}</strong>
                      <small>{evaluationDetails.generalization?.interpretation}</small>
                    </div>
                    <div>
                      <span>Baseline comparison</span>
                      <strong>{formatLabel(evaluationDetails.baseline_comparison?.status)}</strong>
                    </div>
                    <div>
                      <span>Underfitting suspected</span>
                      <strong>
                        {evaluationDetails.underfitting?.underfitting_suspected ? "Yes" : "No"}
                      </strong>
                      <small>{evaluationDetails.underfitting?.interpretation}</small>
                    </div>
                  </div>
                  {evaluationDetails.warnings?.length > 0 && (
                    <ul className="warning-list">
                      {evaluationDetails.warnings.map((warning, index) => (
                        <li key={index}>{warning}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </section>

            <section className="panel">
              <SectionHeading
                eyebrow="MODEL EXPLAINABILITY"
                title="Feature importance"
                description="Features with higher importance contributed more to the model's predictions. Importance does not establish causation."
              />
              {importanceData && (
    <p className="importance-method">
      Calculation method: {importanceMethod.replaceAll("_", " ")}
    </p>
  )}

              {normalizedImportance.length ? (
                <div className="importance-chart">
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={normalizedImportance} layout="vertical" margin={{ left: 25, right: 25 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                      <XAxis type="number" />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        width={125}
                        tick={{ fontSize: 12 }}
                      />
                      <Tooltip />
                      <Bar dataKey="importance" name="Importance" fill="#7568e8" radius={[0, 5, 5, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="empty-state">
                  <BrainCircuit size={28} />
                  <p>Feature importance is not available for the selected model.</p>
                  <small>
                    {explainability?.error ||
                      explainability?.native_importance?.reason ||
                      explainability?.coefficients?.reason ||
                      "This model may not expose native feature importance or coefficients."}
                  </small>
                </div>
              )}
            </section>

            {cleaning && (
              <section className="panel">
                <SectionHeading
                  eyebrow="DATA CLEANING"
                  title="Cleaning summary"
                  description="Summary of the cleaned dataset generated during analysis."
                />

                <div className="metrics-grid">
                  <MetricCard icon={Database} label="Original rows" value={cleaning.original_rows} />
                  <MetricCard icon={CheckCircle2} label="Cleaned rows" value={cleaning.cleaned_rows} />
                  <MetricCard icon={RefreshCw} label="Duplicates removed" value={cleaning.duplicate_rows_removed} />
                  <MetricCard icon={AlertTriangle} label="Remaining missing" value={cleaning.remaining_missing_values} />
                </div>

                {cleaning.actions?.length > 0 && (
                  <ul className="action-list">
                    {cleaning.actions.map((action, index) => (
                      <li key={index}>{action}</li>
                    ))}
                  </ul>
                )}
              </section>
            )}

            <section className="panel" id="reports">
              <SectionHeading
                eyebrow="EXPORT CENTER"
                title="Download your reports"
                description="Save the outputs of your ML diagnostics for further review."
              />

              <div className="download-grid">
                <DownloadButton
                  type="cleaned"
                  filename={result.cleaned_file}
                  label="Cleaned CSV"
                  icon={FileSpreadsheet}
                />
                <DownloadButton
                  type="json"
                  filename={result.reports?.json}
                  label="JSON Report"
                  icon={FileText}
                />
                <DownloadButton
                  type="html"
                  filename={result.reports?.html}
                  label="HTML Report"
                  icon={Download}
                />
              </div>
            </section>

            <p className="disclaimer">
              Model scores are estimates based on the supplied dataset and one
              holdout split. They do not guarantee future performance.
              Validate on representative independent data before deployment.
            </p>
          </>
        )}

        <footer className="footer">
          <strong>ML Model Doctor</strong>
          <span>Clean data. Evaluate honestly. Improve iteratively.</span>
        </footer>
      </main>
    </div>
  );
}

export default App;

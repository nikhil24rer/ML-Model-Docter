
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

async function handleResponse(response) {
  const data = await response.json();

  if (!response.ok || data.status === "error") {
    throw new Error(data.message || "Something went wrong.");
  }

  return data;
}

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/api/upload`, {
    method: "POST",
    body: formData,
  });

  return handleResponse(response);
}

export async function analyzeDataset({
  filename,
  targetColumn,
  problemType,
  maxModels = 6,
  testSize = 0.2,
}) {
  const response = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      filename,
      target_column: targetColumn,
      problem_type: problemType,
      max_models: maxModels,
      test_size: testSize,
    }),
  });

  return handleResponse(response);
}

export function getDownloadUrl(type, filename) {
  const routes = {
    cleaned: "/api/download-cleaned/",
    json: "/api/download-report/json/",
    html: "/api/download-report/html/",
  };

  if (!routes[type] || !filename) {
    throw new Error("Invalid download request.");
  }

  return `${API_URL}${routes[type]}${encodeURIComponent(filename)}`;
}

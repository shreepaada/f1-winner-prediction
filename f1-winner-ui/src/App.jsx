// src/App.jsx
import { useEffect, useState } from "react";
import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:5000";

function App() {
  const [metadata, setMetadata] = useState(null);
  const [metaLoading, setMetaLoading] = useState(true);
  const [metaError, setMetaError] = useState(null);

  const [selectedModel, setSelectedModel] = useState("catboost");
  const [featureValues, setFeatureValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState(null);

  // Load /metadata once on mount
  useEffect(() => {
    const loadMetadata = async () => {
      try {
        setMetaLoading(true);
        setMetaError(null);

        const res = await axios.get(`${API_BASE_URL}/metadata`);
        const data = res.data;

        setMetadata(data);

        // Initialise all features to 0 so user doesn’t type everything from scratch
        if (Array.isArray(data.feature_names)) {
          const initialValues = {};
          data.feature_names.forEach((fname) => {
            initialValues[fname] = 0;
          });
          setFeatureValues(initialValues);
        }
      } catch (err) {
        console.error("Failed to load metadata:", err);
        setMetaError("Failed to load metadata from backend.");
      } finally {
        setMetaLoading(false);
      }
    };

    loadMetadata();
  }, []);

  const handleFeatureChange = (name, value) => {
    // Store as string in state; we’ll convert to number while sending
    setFeatureValues((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!metadata || !metadata.feature_names) return;

    setPredictError(null);
    setPrediction(null);
    setPredicting(true);

    try {
      // Build features payload with correct numeric types
      const featuresPayload = {};
      metadata.feature_names.forEach((fname) => {
        let raw = featureValues[fname];
        if (raw === "" || raw === undefined || raw === null) {
          raw = 0;
        }
        const num = Number(raw);
        featuresPayload[fname] = Number.isNaN(num) ? 0 : num;
      });

      const body = {
        model: selectedModel, // "catboost" or "xgboost"
        features: featuresPayload,
      };

      const res = await axios.post(`${API_BASE_URL}/predict`, body);
      setPrediction(res.data);
    } catch (err) {
      console.error("Prediction failed:", err);
      if (err.response && err.response.data && err.response.data.error) {
        setPredictError(err.response.data.error);
      } else {
        setPredictError("Prediction request failed.");
      }
    } finally {
      setPredicting(false);
    }
  };

  // UI rendering
  if (metaLoading) {
    return (
      <div style={{ padding: "1.5rem", fontFamily: "sans-serif" }}>
        <h2>F1 Winner Prediction</h2>
        <p>Loading metadata from backend…</p>
      </div>
    );
  }

  if (metaError) {
    return (
      <div style={{ padding: "1.5rem", fontFamily: "sans-serif" }}>
        <h2>F1 Winner Prediction</h2>
        <p style={{ color: "red" }}>{metaError}</p>
      </div>
    );
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        padding: "1.5rem",
        fontFamily: "sans-serif",
        backgroundColor: "#0b1020",
        color: "#f5f5f5",
      }}
    >
      <h1 style={{ marginBottom: "0.5rem" }}>F1 Race Winner Predictor</h1>
      <p style={{ marginBottom: "1rem", color: "#ccc" }}>
        Backend: heavy-tuned CatBoost / XGBoost models (ROC-AUC ≈ 0.98+).
      </p>

      {/* Quick meta info cards */}
      <div
        style={{
          display: "flex",
          gap: "1rem",
          marginBottom: "1.5rem",
          flexWrap: "wrap",
        }}
      >
        <div
          style={{
            padding: "0.75rem 1rem",
            borderRadius: "0.5rem",
            background: "#151a30",
          }}
        >
          <div style={{ fontSize: "0.85rem", color: "#aaa" }}>Drivers</div>
          <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>
            {metadata.drivers ? metadata.drivers.length : 0}
          </div>
        </div>
        <div
          style={{
            padding: "0.75rem 1rem",
            borderRadius: "0.5rem",
            background: "#151a30",
          }}
        >
          <div style={{ fontSize: "0.85rem", color: "#aaa" }}>Seasons</div>
          <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>
            {metadata.seasons && metadata.seasons.length > 0
              ? `${metadata.seasons[0]}–${
                  metadata.seasons[metadata.seasons.length - 1]
                }`
              : "N/A"}
          </div>
        </div>
        <div
          style={{
            padding: "0.75rem 1rem",
            borderRadius: "0.5rem",
            background: "#151a30",
          }}
        >
          <div style={{ fontSize: "0.85rem", color: "#aaa" }}>Features</div>
          <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>
            {metadata.n_features}
          </div>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 2fr) minmax(0, 1.2fr)",
          gap: "1.5rem",
        }}
      >
        {/* LEFT: Form */}
        <div
          style={{
            background: "#151a30",
            padding: "1rem",
            borderRadius: "0.75rem",
            border: "1px solid #232949",
          }}
        >
          <h2 style={{ marginBottom: "0.75rem" }}>Input Features</h2>

          {/* Model selector */}
          <div style={{ marginBottom: "1rem" }}>
            <label
              style={{
                display: "block",
                marginBottom: "0.35rem",
                fontSize: "0.9rem",
              }}
            >
              Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                width: "100%",
                padding: "0.4rem 0.5rem",
                borderRadius: "0.4rem",
                border: "1px solid #333a5c",
                background: "#0f1425",
                color: "#f5f5f5",
              }}
            >
              <option value="catboost">CatBoost (tuned, default)</option>
              <option value="xgboost">XGBoost (tuned)</option>
            </select>
          </div>

          {/* Simple helper text */}
          <p style={{ fontSize: "0.85rem", color: "#aaa", marginBottom: "0.75rem" }}>
            All features are initialised to <code>0</code>. Change only the ones
            you care about (e.g. qualifying position, constructor points, etc.),
            then hit <b>Predict</b>.
          </p>

          {/* Feature fields (scrollable) */}
          <form onSubmit={handleSubmit}>
            <div
              style={{
                maxHeight: "380px",
                overflowY: "auto",
                borderRadius: "0.5rem",
                border: "1px solid #232949",
                padding: "0.75rem",
                marginBottom: "1rem",
                background: "#0f1425",
              }}
            >
              {metadata.feature_names.map((fname) => (
                <div
                  key={fname}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                    marginBottom: "0.4rem",
                  }}
                >
                  <label
                    style={{
                      flex: "0 0 55%",
                      fontSize: "0.8rem",
                      color: "#ccc",
                    }}
                  >
                    {fname}
                  </label>
                  <input
                    type="number"
                    step="any"
                    value={
                      featureValues[fname] === undefined
                        ? ""
                        : featureValues[fname]
                    }
                    onChange={(e) =>
                      handleFeatureChange(fname, e.target.value)
                    }
                    style={{
                      flex: "1",
                      padding: "0.25rem 0.4rem",
                      borderRadius: "0.35rem",
                      border: "1px solid #333a5c",
                      background: "#151a30",
                      color: "#f5f5f5",
                      fontSize: "0.8rem",
                    }}
                  />
                </div>
              ))}
            </div>

            <button
              type="submit"
              disabled={predicting}
              style={{
                padding: "0.5rem 1.2rem",
                borderRadius: "0.5rem",
                border: "none",
                background: predicting ? "#444d7a" : "#3b82f6",
                color: "#f5f5f5",
                fontWeight: 600,
                cursor: predicting ? "default" : "pointer",
              }}
            >
              {predicting ? "Predicting…" : "Predict Win Probability"}
            </button>
          </form>

          {predictError && (
            <p style={{ marginTop: "0.75rem", color: "#ff6b6b" }}>
              {predictError}
            </p>
          )}
        </div>

        {/* RIGHT: Prediction + raw metadata helper */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "1rem",
          }}
        >
          <div
            style={{
              background: "#151a30",
              padding: "1rem",
              borderRadius: "0.75rem",
              border: "1px solid #232949",
            }}
          >
            <h2 style={{ marginBottom: "0.75rem" }}>Prediction Result</h2>
            {prediction ? (
              <div>
                <p style={{ marginBottom: "0.4rem" }}>
                  <b>Model used:</b> {prediction.model_used}
                </p>
                <p style={{ marginBottom: "0.4rem" }}>
                  <b>Probability (driver wins race):</b>{" "}
                  {(prediction.probability_win * 100).toFixed(2)}%
                </p>
                <p style={{ marginBottom: "0.4rem" }}>
                  <b>Predicted label:</b>{" "}
                  {prediction.predicted_label === 1 ? "Win" : "Not win"}
                </p>
                <p style={{ fontSize: "0.8rem", color: "#aaa" }}>
                  Threshold: {prediction.threshold}
                </p>
              </div>
            ) : (
              <p style={{ color: "#aaa" }}>
                No prediction yet. Fill some features and click{" "}
                <b>Predict Win Probability</b>.
              </p>
            )}
          </div>

          <div
            style={{
              background: "#151a30",
              padding: "1rem",
              borderRadius: "0.75rem",
              border: "1px solid #232949",
            }}
          >
            <h3 style={{ marginBottom: "0.5rem" }}>Metadata (read-only)</h3>
            <p style={{ fontSize: "0.85rem", color: "#aaa" }}>
              Frontend loaded <b>{metadata.races.length}</b> races from backend.
              Example race:
            </p>
            {metadata.races && metadata.races.length > 0 && (
              <pre
                style={{
                  marginTop: "0.5rem",
                  fontSize: "0.75rem",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  background: "#0f1425",
                  padding: "0.5rem",
                  borderRadius: "0.5rem",
                  border: "1px solid #232949",
                }}
              >
                {JSON.stringify(metadata.races[0], null, 2)}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

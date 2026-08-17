import React from "react";
import ResultCard from "./ResultCard";

export default function ResultGrid({
  results,
  isLoading,
  error,
  latency,
  engineType,
  queryType,
  onSelectResult,
  topK,
}) {
  if (error) {
    return (
      <div className="results-section">
        <div className="error-banner">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <div>
            <strong>Error: </strong> {error}
          </div>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="results-section">
        <div className="results-meta-bar">
          <div className="meta-stats">
            <div className="meta-item">
              <span>Querying vector index...</span>
            </div>
          </div>
        </div>
        <div className="results-grid">
          {Array.from({ length: topK }).map((_, i) => (
            <div key={i} className="skeleton-card">
              <div className="skeleton-shimmer" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className="results-section">
        <div className="empty-state">
          <p style={{ fontSize: "1rem", color: "var(--text-secondary)", marginBottom: "6px" }}>
            Ready to retrieve nearest neighbors
          </p>
          <p style={{ fontSize: "0.85rem" }}>
            Select an image or type a text prompt above to execute vector search across the CUB-200 gallery.
          </p>
        </div>
      </div>
    );
  }

  const engineLabel =
    queryType === "text"
      ? "OpenCLIP ViT-B/32 (512D)"
      : engineType === "resnet"
      ? "ResNet34 Proxy-Anchor (256D)"
      : "OpenCLIP ViT-B/32 (512D)";

  return (
    <div className="results-section">
      <div className="results-meta-bar">
        <div className="meta-stats">
          <div className="meta-item">
            <span>Retrieved: </span>
            <strong className="mono">{results.length} nearest neighbors</strong>
          </div>
          <div className="meta-item">
            <span>Model: </span>
            <strong className="mono">{engineLabel}</strong>
          </div>
        </div>

        {latency !== null && (
          <div className="meta-item mono" style={{ color: "var(--color-accent)" }}>
            <span>Latency: </span>
            <strong>{latency} ms</strong>
          </div>
        )}
      </div>

      <div className="results-grid">
        {results.map((item, idx) => (
          <ResultCard
            key={item.id ?? idx}
            result={item}
            rank={idx + 1}
            onSelect={onSelectResult}
          />
        ))}
      </div>
    </div>
  );
}

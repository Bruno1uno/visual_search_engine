import React, { useEffect } from "react";
import { formatClassName } from "../api";

export default function ImageModal({ selectedItem, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!selectedItem) return null;

  const { result, rank } = selectedItem;
  const imageUrl = `/static/images/${result.rel_path}`;
  const speciesName = formatClassName(result.class_name);
  const scorePercent = (result.score * 100).toFixed(2);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header-row" style={{ padding: "16px 20px", borderBottom: "1px solid var(--border-subtle)" }}>
          <div>
            <span className="card-rank-badge mono" style={{ position: "static", marginRight: "8px" }}>
              Rank #{rank}
            </span>
            <span className="card-score-badge mono" style={{ position: "static" }}>
              Cosine Similarity: {scorePercent}%
            </span>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} title="Close (Esc)">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <img src={imageUrl} alt={speciesName} className="modal-image" />

        <div className="modal-body">
          <div>
            <h3 style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--text-primary)" }}>
              {speciesName}
            </h3>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", background: "rgba(15,23,42,0.6)", padding: "14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Class ID</span>
              <div className="mono" style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--text-primary)" }}>
                #{result.class_id}
              </div>
            </div>
            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Dataset Image ID</span>
              <div className="mono" style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--text-primary)" }}>
                {result.image_id}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

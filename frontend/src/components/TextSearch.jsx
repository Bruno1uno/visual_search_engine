import React from "react";
import { SAMPLE_PROMPTS } from "../api";

export default function TextSearch({
  textQuery,
  setTextQuery,
  onSearch,
  isLoading,
  topK,
  setTopK,
}) {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (textQuery.trim()) {
      onSearch(textQuery.trim(), topK);
    }
  };

  const handleChipClick = (prompt) => {
    setTextQuery(prompt);
    onSearch(prompt, topK);
  };

  return (
    <div className="control-card">
      {/* Search Input Form */}
      <form onSubmit={handleSubmit} className="search-input-wrapper">
        <div className="search-input-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </div>

        <input
          type="text"
          className="search-input"
          placeholder="Describe a bird in English (e.g. 'yellow bird with black wings')..."
          value={textQuery}
          onChange={(e) => setTextQuery(e.target.value)}
          autoFocus
        />

        <button
          type="submit"
          className="search-submit-btn"
          disabled={!textQuery.trim() || isLoading}
        >
          {isLoading ? (
            <span>Searching...</span>
          ) : (
            <>
              <span>Search</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </>
          )}
        </button>
      </form>

      {/* Example Prompt Chips */}
      <div className="sample-section">
        <div className="section-label">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          <span>Try example queries:</span>
        </div>
        <div className="chips-container">
          {SAMPLE_PROMPTS.map((prompt, idx) => (
            <button
              key={idx}
              type="button"
              className="prompt-chip"
              onClick={() => handleChipClick(prompt)}
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Options Bar: Engine Badge & Top-K */}
      <div className="options-bar">
        <div className="option-group">
          <span className="section-label">Engine:</span>
          <div className="status-badge" style={{ padding: "4px 10px" }}>
            <span className="mono" style={{ color: "var(--text-accent)" }}>
              OpenCLIP ViT-B/32 (512D Text-to-Image)
            </span>
          </div>
        </div>

        <div className="option-group">
          <span className="section-label">Top-K:</span>
          <div className="topk-selector">
            {[4, 8, 12, 16, 24].map((k) => (
              <button
                key={k}
                type="button"
                className={`topk-pill ${topK === k ? "active" : ""}`}
                onClick={() => setTopK(k)}
              >
                {k}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

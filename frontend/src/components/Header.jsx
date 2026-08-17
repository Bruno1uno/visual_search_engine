import React from "react";

export default function Header({ health }) {
  const isOnline = health?.engine_ready;
  const numVectors = health?.num_indexed_vectors ?? 0;

  return (
    <header className="header-wrapper">
      <div className="header-content">
        <div className="brand-section">
          <div className="brand-icon-wrapper">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
              <path d="M11 8v6" />
              <path d="M8 11h6" />
            </svg>
          </div>
          <div>
            <span className="brand-title">Visual Search</span>
          </div>
          <span className="brand-badge">CUB-200</span>
        </div>

        <div className="status-badge">
          <span className={`status-dot ${isOnline ? "online" : "offline"}`} />
          <span className="mono">
            {isOnline
              ? `API Connected (${numVectors.toLocaleString()} vectors)`
              : "Backend Disconnected"}
          </span>
        </div>
      </div>
    </header>
  );
}

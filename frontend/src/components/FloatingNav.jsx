import React from "react";

export default function FloatingNav({ activeMode, onModeChange }) {
  return (
    <nav className="nav-pill-container" aria-label="Search Mode Selection">
      <button
        type="button"
        className={`nav-pill-btn ${activeMode === "image" ? "active" : ""}`}
        onClick={() => onModeChange("image")}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
          <circle cx="9" cy="9" r="2" />
          <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
        </svg>
        <span>Image Search</span>
      </button>

      <button
        type="button"
        className={`nav-pill-btn ${activeMode === "text" ? "active" : ""}`}
        onClick={() => onModeChange("text")}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 7V4h16v3" />
          <path d="M9 20h6" />
          <path d="M12 4v16" />
        </svg>
        <span>Text-to-Image</span>
      </button>
    </nav>
  );
}

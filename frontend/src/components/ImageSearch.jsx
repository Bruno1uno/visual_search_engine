import React, { useRef, useState } from "react";
import { SAMPLE_IMAGES } from "../api";

export default function ImageSearch({
  onSearch,
  isLoading,
  selectedFile,
  setSelectedFile,
  previewUrl,
  setPreviewUrl,
  engineType,
  setEngineType,
  topK,
  setTopK,
}) {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      return;
    }
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleSelectSample = async (sample) => {
    try {
      // Fetch sample image and convert to Blob/File
      const res = await fetch(sample.path);
      const blob = await res.blob();
      const filename = sample.path.split("/").pop() || "sample_bird.jpg";
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      setSelectedFile(file);
      setPreviewUrl(sample.path);
    } catch (err) {
      console.error("Failed to load sample image:", err);
    }
  };

  const handleClearImage = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleTriggerSearch = () => {
    if (selectedFile) {
      onSearch(selectedFile, engineType, topK);
    }
  };

  return (
    <div className="control-card">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleFileChange(e.target.files[0]);
          }
        }}
      />

      {/* Upload Dropzone or Preview */}
      {!previewUrl ? (
        <div
          className={`dropzone ${isDragActive ? "drag-active" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="dropzone-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <div className="dropzone-title">Drop your query image here, or browse</div>
          <div className="dropzone-desc">Supports JPEG, PNG, WebP up to 10MB</div>
        </div>
      ) : (
        <div className="preview-container">
          <img src={previewUrl} alt="Query preview" className="preview-thumb" />
          <div className="preview-info">
            <div className="preview-name">{selectedFile?.name || "Sample Image"}</div>
            <div className="preview-size">
              {selectedFile?.size
                ? `${(selectedFile.size / 1024).toFixed(1)} KB`
                : "CUB-200 Sample Dataset"}
            </div>
          </div>
          <button
            type="button"
            className="preview-remove-btn"
            onClick={handleClearImage}
          >
            Change Image
          </button>
        </div>
      )}

      {/* Quick Sample Birds Picker */}
      <div className="sample-section">
        <div className="section-label">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
          </svg>
          <span>Or test with a sample bird:</span>
        </div>
        <div className="sample-grid">
          {SAMPLE_IMAGES.map((sample, idx) => (
            <button
              key={idx}
              type="button"
              className={`sample-chip ${previewUrl === sample.path ? "active" : ""}`}
              onClick={() => handleSelectSample(sample)}
            >
              <img src={sample.path} alt={sample.name} className="sample-chip-img" />
              <span className="sample-chip-text">{sample.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Options Bar: Search Engine & Top-K */}
      <div className="options-bar">
        <div className="option-group">
          <span className="section-label">Model Engine:</span>
          <div className="segmented-toggle">
            <button
              type="button"
              className={`segmented-btn ${engineType === "resnet" ? "active" : ""}`}
              onClick={() => setEngineType("resnet")}
              title="ResNet34 trained with Proxy-Anchor Metric Learning (128D/256D)"
            >
              ResNet34 (Metric Learning)
            </button>
            <button
              type="button"
              className={`segmented-btn ${engineType === "clip" ? "active" : ""}`}
              onClick={() => setEngineType("clip")}
              title="OpenCLIP ViT-B/32 Zero-Shot Embedding (512D)"
            >
              OpenCLIP (Zero-Shot)
            </button>
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

      {/* Primary Action Button */}
      <button
        type="button"
        className="btn-primary-action"
        disabled={!selectedFile || isLoading}
        onClick={handleTriggerSearch}
      >
        {isLoading ? (
          <>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}>
              <line x1="12" y1="2" x2="12" y2="6" />
              <line x1="12" y1="18" x2="12" y2="22" />
              <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
              <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
              <line x1="2" y1="12" x2="6" y2="12" />
              <line x1="18" y1="12" x2="22" y2="12" />
              <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
              <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
            </svg>
            <span>Searching Vector Index...</span>
          </>
        ) : (
          <>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <span>Search Similar Birds</span>
          </>
        )}
      </button>
    </div>
  );
}

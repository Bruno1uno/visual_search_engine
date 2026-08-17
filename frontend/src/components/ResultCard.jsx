import React from "react";
import { formatClassName } from "../api";

export default function ResultCard({ result, rank, onSelect }) {
  const imageUrl = `/static/images/${result.rel_path}`;
  const speciesName = formatClassName(result.class_name);
  // Format score as percentage if in 0..1 range, else 3 decimals
  const scorePercent = (result.score * 100).toFixed(1);

  return (
    <div
      className="result-card"
      onClick={() => onSelect(result, rank)}
      title={`Click to inspect details: ${speciesName}`}
    >
      <div className="card-image-wrap">
        <img
          src={imageUrl}
          alt={speciesName}
          className="card-img"
          loading="lazy"
          onError={(e) => {
            e.currentTarget.src =
              "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' fill='%231e293b'/><text x='50' y='50' fill='%2394a3b8' font-size='12' text-anchor='middle' alignment-baseline='middle'>Image Unavailable</text></svg>";
          }}
        />
        <span className="card-rank-badge mono">#{rank}</span>
        <span className="card-score-badge mono">{scorePercent}%</span>
      </div>

      <div className="card-content">
        <div className="card-species-name" title={speciesName}>
          {speciesName}
        </div>
        <div className="card-id-row mono">
          <span>Class #{result.class_id}</span>
          <span>Img ID: {result.image_id}</span>
        </div>
      </div>
    </div>
  );
}

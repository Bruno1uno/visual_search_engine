/**
 * Visual Search Engine API Service
 * Interacts with FastAPI backend endpoints.
 */

// Sample bird images available in CUB-200-2011 for instant testing
export const SAMPLE_IMAGES = [
  {
    name: "Black-footed Albatross",
    path: "/static/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg",
  },
  {
    name: "Laysan Albatross",
    path: "/static/images/002.Laysan_Albatross/Laysan_Albatross_0001_545.jpg",
  },
  {
    name: "Sooty Albatross",
    path: "/static/images/003.Sooty_Albatross/Sooty_Albatross_0001_1071.jpg",
  },
  {
    name: "Groove-billed Ani",
    path: "/static/images/004.Groove_billed_Ani/Groove_Billed_Ani_0002_1670.jpg",
  },
  {
    name: "Crested Auklet",
    path: "/static/images/005.Crested_Auklet/Crested_Auklet_0001_794941.jpg",
  },
  {
    name: "Least Auklet",
    path: "/static/images/006.Least_Auklet/Least_Auklet_0004_795112.jpg",
  },
];

// Sample prompt queries for Text-to-Image search
export const SAMPLE_PROMPTS = [
  "yellow bird with black wings and black cap",
  "small bright blue bird sitting on tree branch",
  "red crowned woodpecker on a tree trunk",
  "large white seagull with sharp yellow beak",
  "dark brown hawk with spotted breast feathers",
  "bright green hummingbird hovering in air",
];

/**
 * Health check endpoint
 */
export async function checkHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return await response.json();
}

/**
 * Search by image upload
 * @param {File|Blob} imageFile
 * @param {string} engineType - 'resnet' | 'clip'
 * @param {number} topK
 */
export async function searchByImage(imageFile, engineType = "resnet", topK = 8) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const queryParams = new URLSearchParams({
    engine_type: engineType,
    top_k: topK.toString(),
  });

  const response = await fetch(`/api/search/image?${queryParams}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Image search failed: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Search by text prompt
 * @param {string} textQuery
 * @param {number} topK
 */
export async function searchByText(textQuery, topK = 8) {
  const response = await fetch("/api/search/text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text_query: textQuery,
      top_k: topK,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Text search failed: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Clean raw class name e.g. "001.Black_footed_Albatross" -> "Black footed Albatross"
 */
export function formatClassName(rawName) {
  if (!rawName) return "Unknown Species";
  // Remove leading numbers and dot (e.g. 001.)
  const withoutId = rawName.replace(/^\d+\./, "");
  // Replace underscores with spaces
  return withoutId.replace(/_/g, " ");
}

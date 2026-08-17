import React, { useEffect, useState } from "react";
import { checkHealth, searchByImage, searchByText } from "./api";
import Header from "./components/Header";
import Footer from "./components/Footer";
import FloatingNav from "./components/FloatingNav";
import ImageSearch from "./components/ImageSearch";
import TextSearch from "./components/TextSearch";
import ResultGrid from "./components/ResultGrid";
import ImageModal from "./components/ImageModal";

export default function App() {
  const [activeMode, setActiveMode] = useState("image"); // 'image' | 'text'
  const [health, setHealth] = useState(null);

  // Image mode state
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [engineType, setEngineType] = useState("resnet"); // 'resnet' | 'clip'

  // Text mode state
  const [textQuery, setTextQuery] = useState("");

  // Shared state
  const [topK, setTopK] = useState(8);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [latency, setLatency] = useState(null);
  const [selectedModalItem, setSelectedModalItem] = useState(null);

  // Check health on mount and periodically
  useEffect(() => {
    let isMounted = true;
    const fetchHealth = async () => {
      try {
        const data = await checkHealth();
        if (isMounted) setHealth(data);
      } catch (err) {
        if (isMounted) setHealth({ engine_ready: false, num_indexed_vectors: 0 });
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Handler for Image Search
  const handleImageSearch = async (file, engine, k) => {
    setIsLoading(true);
    setError(null);
    const startTime = performance.now();

    try {
      const data = await searchByImage(file, engine, k);
      const elapsed = Math.round(performance.now() - startTime);
      setResults(data.results || []);
      setLatency(elapsed);
    } catch (err) {
      console.error("Image search error:", err);
      setError(err.message || "Failed to execute image search.");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handler for Text Search
  const handleTextSearch = async (query, k) => {
    setIsLoading(true);
    setError(null);
    const startTime = performance.now();

    try {
      const data = await searchByText(query, k);
      const elapsed = Math.round(performance.now() - startTime);
      setResults(data.results || []);
      setLatency(elapsed);
    } catch (err) {
      console.error("Text search error:", err);
      setError(err.message || "Failed to execute text search.");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Mode change handler (clears results if switching)
  const handleModeChange = (mode) => {
    setActiveMode(mode);
    setError(null);
  };

  return (
    <div className="app-layout">
      <Header health={health} />

      <main className="app-container">
        {/* Hero Section */}
        <section className="hero-section">
          <h1 className="hero-title">Visual Search Engine</h1>
          <p className="hero-subtitle">
            Deep Metric Learning (ResNet34 Proxy-Anchor) and Multi-Modal Zero-Shot Retrieval (OpenCLIP) on CUB-200-2011
          </p>
        </section>

        {/* Floating Centered Pill Navigation */}
        <FloatingNav activeMode={activeMode} onModeChange={handleModeChange} />

        {/* Retrieval Control Panel */}
        {activeMode === "image" ? (
          <ImageSearch
            onSearch={handleImageSearch}
            isLoading={isLoading}
            selectedFile={selectedFile}
            setSelectedFile={setSelectedFile}
            previewUrl={previewUrl}
            setPreviewUrl={setPreviewUrl}
            engineType={engineType}
            setEngineType={setEngineType}
            topK={topK}
            setTopK={setTopK}
          />
        ) : (
          <TextSearch
            textQuery={textQuery}
            setTextQuery={setTextQuery}
            onSearch={handleTextSearch}
            isLoading={isLoading}
            topK={topK}
            setTopK={setTopK}
          />
        )}

        {/* Results Grid */}
        <ResultGrid
          results={results}
          isLoading={isLoading}
          error={error}
          latency={latency}
          engineType={engineType}
          queryType={activeMode}
          onSelectResult={(result, rank) => setSelectedModalItem({ result, rank })}
          topK={topK}
        />
      </main>

      <Footer />

      {/* Lightbox Modal */}
      <ImageModal
        selectedItem={selectedModalItem}
        onClose={() => setSelectedModalItem(null)}
      />
    </div>
  );
}

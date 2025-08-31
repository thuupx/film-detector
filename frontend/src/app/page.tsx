"use client";

import { useState, useRef } from "react";
import Image from "next/image";
import { FilmPredictor } from "../components/FilmPredictor";

const predictor = new FilmPredictor();


export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setPredictions(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError(null);

    try {
      const results = await predictor.predict(selectedImage);
      setPredictions(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setPredictions(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-800 dark:text-slate-200 mb-2">
            Film Settings Detector
          </h1>
          <p className="text-slate-600 dark:text-slate-400 text-lg">
            Upload a photo to predict optimal film simulation settings
          </p>
        </header>

        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Image Upload Section */}
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-slate-800 dark:text-slate-200 mb-4">
                Upload Image
              </h2>
              
              <div className="space-y-4">
                <div className="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-lg p-6 text-center">
                  {imagePreview ? (
                    <div className="space-y-4">
                      <div className="relative aspect-video max-w-sm mx-auto">
                        <Image
                          src={imagePreview}
                          alt="Preview"
                          fill
                          className="object-contain rounded-lg"
                        />
                      </div>
                      <div className="flex gap-2 justify-center">
                        <button
                          onClick={handlePredict}
                          disabled={isLoading}
                          className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-lg font-medium transition-colors"
                        >
                          {isLoading ? "Analyzing..." : "Predict Settings"}
                        </button>
                        <button
                          onClick={handleClear}
                          className="px-6 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg font-medium transition-colors"
                        >
                          Clear
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="text-slate-600 dark:text-slate-400">
                        <p className="text-lg mb-2">Choose an image to analyze</p>
                        <p className="text-sm">Supports JPG, PNG, and WEBP formats</p>
                      </div>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
                      >
                        Select Image
                      </button>
                    </div>
                  )}
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                />
              </div>

              {error && (
                <div className="mt-4 p-4 bg-red-100 dark:bg-red-900/20 border border-red-300 dark:border-red-800 rounded-lg">
                  <p className="text-red-700 dark:text-red-300">{error}</p>
                </div>
              )}
            </div>

            {/* Results Section */}
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-slate-800 dark:text-slate-200 mb-4">
                Predicted Settings
              </h2>
              
              {predictions ? (
                <div className="space-y-4">
                  <div className="grid gap-3">
                    {Object.entries(predictions).map(([key, value]) => (
                      <div key={key} className="flex justify-between items-center py-2 border-b border-slate-200 dark:border-slate-700 last:border-b-0">
                        <span className="font-medium text-slate-700 dark:text-slate-300">
                          {key.replace(/([A-Z])/g, ' $1').trim()}
                        </span>
                        <span className="text-slate-900 dark:text-slate-100 font-mono">
                          {String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center text-slate-500 dark:text-slate-400 py-8">
                  <p>Upload and analyze an image to see predictions</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

"use client"

import type React from "react"
import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Upload, ArrowLeft, Loader2 } from "lucide-react"
import { detectDisease } from "@/api/detect"

interface DetectionResult {
  result: string
  class: string
}

const DISEASES = [
  "1. Eczema - 1677",
  "10. Warts Molluscum and other Viral Infections - 2103",
  "2. Melanoma - 15.75k",
  "3. Atopic Dermatitis - 1.25k",
  "4. Basal Cell Carcinoma (BCC) - 3323",
  "5. Melanocytic Nevi (NV) - 7970",
  "6. Benign Keratosis-like Lesions (BKL) - 2624",
  "7. Psoriasis pictures Lichen Planus and related diseases - 2k",
  "8. Seborrheic Keratoses and other Benign Tumors - 1.8k",
  "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k",
]

export default function DetectionPage() {
  const navigate = useNavigate()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file")
      return
    }

    setSelectedFile(file)
    setError(null)
    setResult(null)

    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.currentTarget.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleDetect = async () => {
    if (!selectedFile) {
      setError("Please select an image first")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const data = await detectDisease(selectedFile)
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during detection")
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border">
        <nav className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <button
            onClick={() => navigate("/")}
            className="flex items-center gap-2 text-primary hover:text-primary/80 transition"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back</span>
          </button>
          <div className="text-xl font-bold text-primary">DermAI</div>
          <div className="w-16" /> {/* Spacer for alignment */}
        </nav>
      </header>

      {/* Main Content */}
      <section className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-20">
        <div className="text-center mb-12">
          <h1 className="text-3xl md:text-4xl font-bold mb-3">Skin Disease Detection</h1>
          <p className="text-muted-foreground text-lg">Upload an image of your skin condition for AI analysis</p>
        </div>

        {!result ? (
          <div className="space-y-6">
            {/* Upload Area */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition cursor-pointer bg-secondary/30"
            >
              <input type="file" id="file-input" hidden onChange={handleInputChange} accept="image/*" />
              <label htmlFor="file-input" className="cursor-pointer">
                <Upload className="w-12 h-12 mx-auto text-muted-foreground mb-3" />
                <p className="text-lg font-medium mb-1">Drag and drop your image here</p>
                <p className="text-muted-foreground">or click to select a file</p>
              </label>
            </div>

            {/* Preview */}
            {preview && (
              <div className="space-y-4">
                <div className="border border-border rounded-lg overflow-hidden bg-card">
                  <img
                    src={preview || "/placeholder.svg"}
                    alt="Preview"
                    className="w-full h-auto max-h-96 object-cover"
                  />
                </div>
                <p className="text-sm text-muted-foreground text-center">Selected: {selectedFile?.name}</p>

                {/* Error Message */}
                {error && (
                  <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg text-destructive text-sm">
                    {error}
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-3 justify-center">
                  <Button variant="outline" onClick={handleReset}>
                    Choose Different Image
                  </Button>
                  <Button
                    size="lg"
                    onClick={handleDetect}
                    disabled={loading}
                    className="bg-primary hover:bg-primary/90 text-primary-foreground"
                  >
                    {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    {loading ? "Analyzing..." : "Analyze Image"}
                  </Button>
                </div>
              </div>
            )}

            {/* Error without preview */}
            {error && !preview && (
              <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg text-destructive text-sm">
                {error}
              </div>
            )}
          </div>
        ) : (
          <ResultDisplay result={result} onReset={handleReset} />
        )}
      </section>
    </main>
  )
}

function ResultDisplay({ result, onReset }: { result: DetectionResult; onReset: () => void }) {
  const isSeriousCondition =
    result.result.toLowerCase().includes("melanoma") || result.result.toLowerCase().includes("basal cell carcinoma")

  return (
    <div className="space-y-8 animate-in fade-in">
      {/* Result Card */}
      <div
        className={`border rounded-lg p-8 ${isSeriousCondition ? "border-destructive/50 bg-destructive/5" : "border-primary/50 bg-primary/5"}`}
      >
        <div className="text-center space-y-4">
          <h2 className="text-2xl font-bold">Detection Result</h2>

          <div
            className={`inline-block px-4 py-2 rounded-full text-sm font-semibold ${isSeriousCondition ? "bg-destructive/20 text-destructive" : "bg-primary/20 text-primary"}`}
          >
            {result.class}
          </div>

          <p className="text-xl font-semibold text-foreground">{result.result}</p>

          {isSeriousCondition && (
            <div className="mt-6 p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
              <p className="text-sm text-destructive font-medium">Important Notice:</p>
              <p className="text-sm text-destructive/80 mt-1">
                This condition may require immediate professional evaluation. Please consult with a dermatologist as
                soon as possible.
              </p>
            </div>
          )}

          <div className="mt-6 p-4 bg-secondary rounded-lg border border-border">
            <p className="text-sm text-muted-foreground">
              This AI-powered analysis is for informational purposes only and should not replace professional medical
              advice. Always consult a qualified dermatologist for proper diagnosis and treatment.
            </p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 justify-center flex-wrap">
        <Button variant="outline" size="lg" onClick={onReset}>
          Analyze Another Image
        </Button>
        <Button size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground">
          Learn More About Treatment
        </Button>
      </div>
    </div>
  )
}

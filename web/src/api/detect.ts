const DISEASES = [
  "1. Eczema - 1677",
  "2. Melanoma - 15.75k",
  "3. Atopic Dermatitis - 1.25k",
  "4. Basal Cell Carcinoma (BCC) - 3323",
  "5. Melanocytic Nevi (NV) - 7970",
  "6. Benign Keratosis-like Lesions (BKL) - 2624",
  "7. Psoriasis pictures Lichen Planus and related diseases - 2k",
  "8. Seborrheic Keratoses and other Benign Tumors - 1.8k",
  "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k",
  "10. Warts Molluscum and other Viral Infections - 2103",
]

export async function detectDisease(
  file: File
): Promise<{ result: string; class: string }> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch("http://localhost:5000/predict", {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    throw new Error("Failed to detect disease")
  }

  const data: { prediction: string } = await response.json()

  // Backend already returns class name
  const disease =
    DISEASES.find(d => d.includes(data.prediction)) ??
    data.prediction

  return {
    result: disease,
    class: "Disease Detected",
  }
}

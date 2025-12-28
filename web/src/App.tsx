import { Routes, Route } from "react-router-dom"
import LandingPage from "./pages/LandingPage"
import DetectionPage from "./pages/DetectionPage"

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/detect" element={<DetectionPage />} />
    </Routes>
  )
}

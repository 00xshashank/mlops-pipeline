import { useNavigate } from "react-router-dom"
import { Check, Shield, Zap } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function LandingPage() {
  const navigate = useNavigate()

  const handleCheckNow = () => {
    navigate("/detect")
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border">
        <nav className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="text-xl font-bold text-primary">DermAI</div>
          <div className="flex gap-8 items-center">
            <a href="#features" className="text-muted-foreground hover:text-foreground transition">
              Features
            </a>
            <a href="#about" className="text-muted-foreground hover:text-foreground transition">
              About
            </a>
          </div>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-32">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div className="space-y-6">
            <div className="inline-block px-3 py-1 rounded-full bg-secondary text-secondary-foreground text-sm font-medium">
              AI-Powered Diagnosis
            </div>

            <h1 className="text-4xl md:text-5xl font-bold leading-tight text-balance">
              Detect skin diseases with confidence
            </h1>

            <p className="text-lg text-muted-foreground leading-relaxed">
              Upload a photo of your skin condition and get instant AI-powered analysis. Receive accurate identification
              of common dermatological conditions in seconds.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <Button
                size="lg"
                className="bg-primary hover:bg-primary/90 text-primary-foreground w-full sm:w-auto"
                onClick={handleCheckNow}
              >
                Check Now
              </Button>
              <Button variant="outline" size="lg" className="w-full sm:w-auto bg-transparent">
                Learn More
              </Button>
            </div>
          </div>

          {/* Right Content - Feature Image */}
          <div className="bg-secondary rounded-lg p-8 flex items-center justify-center min-h-96">
            <div className="text-center">
              <Shield className="w-24 h-24 mx-auto text-primary mb-4 opacity-80" />
              <p className="text-muted-foreground">Secure and accurate analysis</p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="bg-secondary py-20 md:py-32">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Why Choose DermAI?</h2>
            <p className="text-muted-foreground text-lg">State-of-the-art technology for skin health</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                icon: Zap,
                title: "Instant Results",
                description: "Get analysis in seconds with our advanced AI model",
              },
              {
                icon: Shield,
                title: "Privacy Protected",
                description: "Your images are never stored or shared with third parties",
              },
              {
                icon: Check,
                title: "Highly Accurate",
                description: "Trained on thousands of dermatological images",
              },
            ].map((feature, idx) => {
              const Icon = feature.icon
              return (
                <div
                  key={idx}
                  className="bg-background p-6 rounded-lg border border-border hover:border-primary/50 transition"
                >
                  <Icon className="w-10 h-10 text-primary mb-4" />
                  <h3 className="font-semibold text-lg mb-2">{feature.title}</h3>
                  <p className="text-muted-foreground">{feature.description}</p>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-32">
        <div className="text-center space-y-6">
          <h2 className="text-3xl md:text-4xl font-bold">Ready to analyze your skin condition?</h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Start your free assessment today. No registration required.
          </p>
          <Button size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground" onClick={handleCheckNow}>
            Begin Detection
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-secondary py-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-muted-foreground">
            DermAI © 2025. For educational purposes only. Always consult a dermatologist for medical advice.
          </p>
        </div>
      </footer>
    </main>
  )
}

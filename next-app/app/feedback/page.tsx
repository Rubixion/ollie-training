import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { BGPattern } from "@/components/bg-pattern"
import { FeedbackSection } from "@/components/feedback-section"

export const metadata = {
  title: "Feedback â€” Ollie",
  description: "Help train the Ollie model by correcting face match results.",
}

export default function FeedbackPage() {
  return (
    <main className="relative min-h-screen bg-transparent">
      <BGPattern variant="dots" mask="fade-edges" fill="rgba(0,212,255,0.08)" size={24} className="fixed" />
      <Nav />
      <div className="pt-24">
        <FeedbackSection />
      </div>
      <Footer />
    </main>
  )
}

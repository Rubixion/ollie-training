import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { BGPattern } from "@/components/bg-pattern"
import { FeedbackSection } from "@/components/feedback-section"

export const metadata = {
  title: "Feedback - Ollie",
  description: "Help train the Ollie model by correcting face match results.",
}

export default function FeedbackPage() {
  return (
    <main className="relative min-h-screen bg-transparent">
      <BGPattern variant="dots" mask="fade-y" fill="rgba(255,255,255,0.18)" size={28} className="fixed" />
      <Nav />
      <div className="pt-24">
        <FeedbackSection />
      </div>
      <Footer />
    </main>
  )
}


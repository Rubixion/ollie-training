import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { DottedSurface } from "@/components/ui/dotted-surface"
import { FeedbackSection } from "@/components/feedback-section"

export const metadata = {
  title: "Feedback — Ollie",
  description: "Help train the Ollie model by correcting face match results.",
}

export default function FeedbackPage() {
  return (
    <main className="relative min-h-screen bg-black">
      <DottedSurface />
      <Nav />
      <div className="pt-24">
        <FeedbackSection />
      </div>
      <Footer />
    </main>
  )
}

import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { BGPattern } from "@/components/bg-pattern"
import { CelebrityFinder } from "@/components/celebrity-finder"

export const metadata = {
  title: "Find Your Celebrity Match â€” Ollie",
  description: "Upload your photo and discover which celebrity you resemble using Ollie's AI face matching.",
}

export default function FindPage() {
  return (
    <main className="relative min-h-screen bg-transparent">
      <BGPattern variant="grid" mask="fade-edges" fill="rgba(0,212,255,0.05)" size={32} className="fixed" />
      <Nav />
      <div className="pt-24">
        <CelebrityFinder />
      </div>
      <Footer />
    </main>
  )
}

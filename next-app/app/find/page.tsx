import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { DottedSurface } from "@/components/ui/dotted-surface"
import { CelebrityFinder } from "@/components/celebrity-finder"

export const metadata = {
  title: "Find Your Celebrity Match — Ollie",
  description: "Upload your photo and discover which celebrity you resemble using Ollie's AI face matching.",
}

export default function FindPage() {
  return (
    <main className="relative min-h-screen bg-black">
      <DottedSurface />
      <Nav />
      <div className="pt-24">
        <CelebrityFinder />
      </div>
      <Footer />
    </main>
  )
}

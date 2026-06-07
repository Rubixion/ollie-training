import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { BGPattern } from "@/components/bg-pattern"
import { FindPerson } from "@/components/find-person"

export const metadata = {
  title: "Face Search Ollie",
  description: "Internet-scale face search. Coming soon.",
}

export default function FindPersonPage() {
  return (
    <main className="relative min-h-screen bg-transparent">
      <BGPattern variant="diagonal-stripes" mask="fade-edges" fill="rgba(255,255,255,0.04)" size={28} className="fixed" />
      <Nav />
      <div className="pt-24">
        <FindPerson />
      </div>
      <Footer />
    </main>
  )
}

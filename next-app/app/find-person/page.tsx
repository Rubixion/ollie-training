import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { DottedSurface } from "@/components/ui/dotted-surface"
import { FindPerson } from "@/components/find-person"

export const metadata = {
  title: "Find Person — Ollie",
  description: "Internet-scale face search. Coming soon.",
}

export default function FindPersonPage() {
  return (
    <main className="relative min-h-screen bg-black">
      <DottedSurface />
      <Nav />
      <div className="pt-24">
        <FindPerson />
      </div>
      <Footer />
    </main>
  )
}

"use client"

import { ExternalLink } from "lucide-react"

export function Footer() {
  return (
    <footer className="border-t border-white/5 py-10 px-6">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Logo */}
        <div className="flex flex-col items-center md:items-start gap-1">
          <span className="text-white font-black text-lg tracking-widest">OLLIE</span>
          <span className="text-white/25 text-xs">Face recognition AI. Not affiliated with any celebrity.</span>
        </div>

        {/* Center note */}
        <p className="text-white/20 text-xs text-center">
          © 2026 Ollie. Built with a Siamese Neural Network.
        </p>

        {/* GitHub */}
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-white/30 hover:text-white/60 transition-colors text-xs"
          aria-label="GitHub"
        >
          <ExternalLink size={16} />
          <span>GitHub</span>
        </a>
      </div>
    </footer>
  )
}

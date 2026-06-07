"use client"

import { useState, useEffect, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Menu, X, User, LogOut } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useAuth } from "@/components/auth-provider"

const links = [
  { label: "Celebrity Twins", href: "/match" },
  { label: "Face Search", href: "/search" },
  { label: "How It Works", href: "/ai" },
  { label: "Feedback", href: "/feedback" },
  { label: "About", href: "/about" },
]

function ProfileMenu() {
  const { user, signOut, openModal } = useAuth()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [])

  if (!user) {
    return (
      <button
        onClick={() => openModal()}
        className="hidden md:flex items-center gap-2 text-sm font-semibold px-4 py-2 rounded-full border border-white/15 text-white/60 hover:text-white hover:border-white/30 transition-all"
      >
        <User size={14} />
        Sign In
      </button>
    )
  }

  return (
    <div ref={ref} className="hidden md:block relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-9 h-9 rounded-full border border-white/10 bg-white/[0.06] hover:bg-white/[0.10] hover:border-white/25 transition-all flex items-center justify-center"
        aria-label="Account menu"
      >
        <User size={16} className="text-white/50" />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 6 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 6 }}
            transition={{ type: "spring", bounce: 0.15, duration: 0.25 }}
            className="absolute right-0 top-full mt-2 w-52 bg-[#0a0a0a] border border-white/10 rounded-2xl shadow-2xl shadow-black overflow-hidden z-50"
          >
            <div className="px-4 py-3 border-b border-white/5">
              <p className="text-white/25 text-[10px] uppercase tracking-widest mb-0.5">Signed in as</p>
              <p className="text-white/70 text-xs truncate font-medium">{user.email}</p>
            </div>
            <div className="p-1.5">
              <button
                onClick={() => { signOut(); setOpen(false) }}
                className="flex items-center gap-2.5 w-full px-3 py-2.5 rounded-xl text-red-400/70 hover:text-red-400 hover:bg-red-500/8 text-sm transition-all"
              >
                <LogOut size={14} />
                Sign out
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export function Nav() {
  const { user, openModal, signOut } = useAuth()
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20)
    window.addEventListener("scroll", handler, { passive: true })
    return () => window.removeEventListener("scroll", handler)
  }, [])

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-black/60 backdrop-blur-xl border-b border-white/5 shadow-lg shadow-black/40"
          : "bg-transparent"
      }`}
    >
      <nav className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link
          href="/"
          className="text-white font-black text-xl tracking-widest hover:text-white/60 transition-colors"
        >
          OLLIE
        </Link>

        {/* Desktop links */}
        <ul className="hidden md:flex items-center gap-8">
          {links.map((link) => {
            const isActive = pathname === link.href
            return (
              <li key={link.href}>
                <Link
                  href={link.href}
                  aria-current={isActive ? "page" : undefined}
                  className={`text-sm transition-colors ${
                    isActive
                      ? "text-white font-semibold"
                      : "text-white/40 hover:text-white"
                  }`}
                >
                  {link.label}
                </Link>
              </li>
            )
          })}
        </ul>

        {/* Desktop right: profile or sign in */}
        <ProfileMenu />

        {/* Mobile hamburger */}
        <button
          className="md:hidden text-white/80 hover:text-white transition-colors"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </nav>

      {/* Mobile dropdown */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="md:hidden bg-black/90 backdrop-blur-xl border-b border-white/5 px-6 pb-4"
          >
            <ul className="flex flex-col gap-1 pt-2">
              {links.map((link) => {
                const isActive = pathname === link.href
                return (
                  <li key={link.href}>
                    <Link
                      href={link.href}
                      onClick={() => setOpen(false)}
                      className={`block w-full py-3 text-sm transition-colors border-b border-white/5 last:border-0 ${
                        isActive ? "text-white font-semibold" : "text-white/40 hover:text-white"
                      }`}
                    >
                      {link.label}
                    </Link>
                  </li>
                )
              })}
              <li className="pt-2">
                {user ? (
                  <div className="flex flex-col gap-2">
                    <p className="text-white/25 text-xs px-1">{user.email}</p>
                    <button
                      onClick={() => { signOut(); setOpen(false) }}
                      className="flex items-center gap-2 w-full text-sm font-semibold py-3 px-4 rounded-full border border-red-500/20 text-red-400/70 hover:text-red-400 transition-colors"
                    >
                      <LogOut size={14} />
                      Sign out
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => { openModal(); setOpen(false) }}
                    className="flex items-center justify-center gap-2 w-full text-sm font-semibold py-3 rounded-full border border-white/15 text-white/70 hover:text-white transition-colors"
                  >
                    <User size={14} />
                    Sign In / Create Account
                  </button>
                )}
              </li>
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}

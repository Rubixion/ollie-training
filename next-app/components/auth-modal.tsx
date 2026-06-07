"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { X, Eye, EyeOff, Mail, Lock, AlertCircle } from "lucide-react"
import { useAuth } from "@/components/auth-provider"

type Tab = "signin" | "signup"

export function AuthModal() {
  const { isModalOpen, closeModal, signIn, signUp, signInWithGoogle } = useAuth()
  const [tab, setTab] = useState<Tab>("signin")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const switchTab = (t: Tab) => {
    setTab(t)
    setError(null)
    setSuccess(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(null)

    const err = tab === "signin"
      ? await signIn(email, password)
      : await signUp(email, password)

    if (err) {
      setError(err)
    } else if (tab === "signup") {
      setSuccess("Check your email to confirm your account, then sign in.")
    }

    setLoading(false)
  }

  return (
    <AnimatePresence>
      {isModalOpen && (
        <>
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/75 backdrop-blur-sm z-[100]"
            onClick={closeModal}
          />

          <div className="fixed inset-0 z-[101] flex items-center justify-center px-4 pointer-events-none">
            <motion.div
              key="modal"
              initial={{ opacity: 0, scale: 0.96, y: 12 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 12 }}
              transition={{ type: "spring", bounce: 0.18, duration: 0.4 }}
              className="w-full max-w-sm pointer-events-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="bg-[#080808] border border-white/10 rounded-2xl shadow-2xl shadow-black overflow-hidden">

                {/* Header */}
                <div className="flex items-start justify-between px-6 pt-6 pb-4">
                  <div>
                    <h2 className="text-lg font-black text-white tracking-tight">
                      {tab === "signin" ? "Welcome back" : "Join Ollie"}
                    </h2>
                    <p className="text-white/35 text-xs mt-1 leading-relaxed">
                      {tab === "signin"
                        ? "Sign in to search and track your matches."
                        : "Free forever. Save results. Help train the AI."}
                    </p>
                  </div>
                  <button
                    onClick={closeModal}
                    className="text-white/25 hover:text-white/60 transition-colors ml-4 mt-0.5 shrink-0"
                    aria-label="Close"
                  >
                    <X size={17} />
                  </button>
                </div>

                {/* Tabs */}
                <div className="px-6 pb-4">
                  <div className="flex gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/8">
                    {(["signin", "signup"] as const).map((t) => (
                      <button
                        key={t}
                        onClick={() => switchTab(t)}
                        className={`relative flex-1 py-2 rounded-lg text-xs font-bold transition-all ${
                          tab === t ? "text-black" : "text-white/35 hover:text-white/60"
                        }`}
                      >
                        {tab === t && (
                          <motion.div
                            layoutId="auth-tab"
                            className="absolute inset-0 bg-(--ollie-cyan) rounded-lg"
                            transition={{ type: "spring", bounce: 0.2, duration: 0.35 }}
                          />
                        )}
                        <span className="relative z-10">{t === "signin" ? "Sign In" : "Sign Up"}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="px-6 pb-6 space-y-3">
                  {/* Google */}
                  <button
                    onClick={signInWithGoogle}
                    className="w-full flex items-center justify-center gap-2.5 py-2.5 rounded-xl border border-white/10 bg-white/[0.03] text-white/60 text-sm font-semibold hover:bg-white/[0.06] hover:text-white hover:border-white/20 transition-all"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src="https://www.google.com/favicon.ico" alt="" className="w-3.5 h-3.5" />
                    Continue with Google
                  </button>

                  {/* Divider */}
                  <div className="relative flex items-center">
                    <span className="flex-1 border-t border-white/8" />
                    <span className="px-3 text-white/20 text-[10px] uppercase tracking-widest">or</span>
                    <span className="flex-1 border-t border-white/8" />
                  </div>

                  {/* Email + password form */}
                  <form onSubmit={handleSubmit} className="space-y-2.5">
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-white/20 pointer-events-none" size={14} />
                      <input
                        type="email"
                        placeholder="email@example.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                        autoComplete="email"
                        className="w-full bg-white/[0.04] border border-white/10 text-white text-sm rounded-xl pl-9 pr-4 py-2.5 placeholder:text-white/20 focus:outline-none focus:border-white/25 transition-colors"
                      />
                    </div>

                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-white/20 pointer-events-none" size={14} />
                      <input
                        type={showPassword ? "text" : "password"}
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                        minLength={6}
                        autoComplete={tab === "signin" ? "current-password" : "new-password"}
                        className="w-full bg-white/[0.04] border border-white/10 text-white text-sm rounded-xl pl-9 pr-10 py-2.5 placeholder:text-white/20 focus:outline-none focus:border-white/25 transition-colors"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-white/20 hover:text-white/50 transition-colors"
                        tabIndex={-1}
                      >
                        {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                      </button>
                    </div>

                    {error && (
                      <div className="flex items-start gap-2 p-3 rounded-xl bg-red-500/10 border border-red-500/15">
                        <AlertCircle size={13} className="text-red-400 shrink-0 mt-0.5" />
                        <p className="text-red-300 text-xs leading-relaxed">{error}</p>
                      </div>
                    )}

                    {success && (
                      <div className="p-3 rounded-xl bg-green-500/10 border border-green-500/15">
                        <p className="text-green-300 text-xs leading-relaxed">{success}</p>
                      </div>
                    )}

                    <button
                      type="submit"
                      disabled={loading}
                      className="w-full py-2.5 rounded-xl bg-(--ollie-cyan) text-black font-bold text-sm hover:opacity-90 active:scale-[0.98] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      {loading
                        ? (tab === "signin" ? "Signing in..." : "Creating account...")
                        : (tab === "signin" ? "Sign In" : "Create Account")}
                    </button>
                  </form>

                  {tab === "signup" && (
                    <p className="text-white/18 text-[10px] text-center leading-relaxed">
                      By signing up you agree to our{" "}
                      <a href="/terms" className="underline hover:text-white/35 transition-colors">Terms</a>
                      {" "}&amp;{" "}
                      <a href="/privacy" className="underline hover:text-white/35 transition-colors">Privacy Policy</a>
                    </p>
                  )}
                </div>
              </div>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  )
}

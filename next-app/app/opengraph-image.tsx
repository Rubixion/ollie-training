import { ImageResponse } from "next/og"

export const runtime = "edge"
export const size = { width: 1200, height: 630 }
export const contentType = "image/png"

export default function OGImage() {
  return new ImageResponse(
    <div
      style={{
        background: "#000000",
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "sans-serif",
        position: "relative",
      }}
    >
      {/* Grid pattern */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage:
            "radial-gradient(circle, rgba(255,255,255,0.06) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      {/* Glow */}
      <div
        style={{
          position: "absolute",
          width: 600,
          height: 400,
          borderRadius: "50%",
          background: "radial-gradient(ellipse, rgba(34,211,238,0.12) 0%, transparent 70%)",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
        }}
      />

      {/* Content */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 24, position: "relative" }}>
        <div
          style={{
            fontSize: 96,
            fontWeight: 900,
            color: "#ffffff",
            letterSpacing: "0.15em",
            lineHeight: 1,
          }}
        >
          OLLIE
        </div>
        <div
          style={{
            fontSize: 26,
            color: "rgba(255,255,255,0.45)",
            letterSpacing: "0.2em",
            textTransform: "uppercase",
          }}
        >
          AI-powered celebrity face matching
        </div>
        <div
          style={{
            marginTop: 12,
            paddingLeft: 28,
            paddingRight: 28,
            paddingTop: 14,
            paddingBottom: 14,
            borderRadius: 999,
            background: "#22d3ee",
            color: "#000",
            fontWeight: 700,
            fontSize: 20,
            letterSpacing: "0.05em",
          }}
        >
          Find your celebrity lookalike →
        </div>
      </div>

      {/* Bottom stat strip */}
      <div
        style={{
          position: "absolute",
          bottom: 48,
          display: "flex",
          gap: 60,
        }}
      >
        {[
          ["83.5%", "Accuracy"],
          ["9,131", "Celebrities"],
          ["3.3M", "Training Faces"],
        ].map(([val, label]) => (
          <div key={label} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
            <span style={{ color: "#22d3ee", fontWeight: 800, fontSize: 28 }}>{val}</span>
            <span style={{ color: "rgba(255,255,255,0.3)", fontSize: 14, letterSpacing: "0.1em" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>,
    { ...size }
  )
}

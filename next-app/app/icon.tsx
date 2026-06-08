import { ImageResponse } from "next/og"

export const runtime = "edge"
export const size = { width: 32, height: 32 }
export const contentType = "image/png"

export default function Icon() {
  return new ImageResponse(
    <div
      style={{
        background: "#000",
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 6,
      }}
    >
      <div
        style={{
          color: "#22d3ee",
          fontWeight: 900,
          fontSize: 20,
          fontFamily: "sans-serif",
          letterSpacing: "-0.02em",
        }}
      >
        O
      </div>
    </div>,
    { ...size }
  )
}

import { NextRequest, NextResponse } from "next/server"
import { getAuthUser } from "@/lib/auth-server"
import { checkRateLimit, getIp } from "@/lib/rate-limit"

const MAX_IMAGE_BYTES = 7 * 1024 * 1024

export async function POST(req: NextRequest) {
  try {
    const user = await getAuthUser(req)
    if (!user) {
      return NextResponse.json({ error: "Authentication required" }, { status: 401 })
    }

    const ip = getIp(req)
    if (!checkRateLimit(`search:${ip}`, 10, 60_000)) {
      return NextResponse.json({ error: "Too many requests. Please wait a moment." }, { status: 429 })
    }

    const body = await req.json()
    const { image, mode } = body

    if (!image) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 })
    }

    if (typeof image !== "string" || image.length > MAX_IMAGE_BYTES) {
      return NextResponse.json({ error: "Image too large. Max 5 MB." }, { status: 413 })
    }

    if (!image.startsWith("data:image/")) {
      return NextResponse.json({ error: "Invalid image format." }, { status: 400 })
    }

    const gradioPayload = {
      fn_index: 0,
      data: [
        { data: image, mime_type: "image/jpeg" },
        mode ?? "CNN + Features",
      ],
    }

    let gradioRes: Response
    try {
      gradioRes = await fetch(`${process.env.GRADIO_URL ?? "http://127.0.0.1:7860"}/run/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(gradioPayload),
        signal: AbortSignal.timeout(30000),
      })
    } catch {
      return NextResponse.json(
        { error: "Gradio server is not running. Please start the backend." },
        { status: 503 }
      )
    }

    if (!gradioRes.ok) {
      return NextResponse.json({ error: "Backend error. Please try again." }, { status: 502 })
    }

    const result = await gradioRes.json()
    return NextResponse.json(result)
  } catch (err) {
    console.error("Search API error:", err)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

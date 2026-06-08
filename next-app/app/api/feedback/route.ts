import { NextRequest, NextResponse } from "next/server"
import { getAuthUser } from "@/lib/auth-server"
import { checkRateLimit, getIp } from "@/lib/rate-limit"

export async function POST(req: NextRequest) {
  try {
    const user = await getAuthUser(req)
    if (!user) {
      return NextResponse.json({ error: "Authentication required" }, { status: 401 })
    }

    const ip = getIp(req)
    if (!checkRateLimit(`feedback:${ip}`, 20, 60_000)) {
      return NextResponse.json({ error: "Too many requests. Please wait a moment." }, { status: 429 })
    }

    const body = await req.json()
    const { image1, image2, label } = body

    if (image1 === undefined || image2 === undefined || label === undefined) {
      return NextResponse.json({ error: "Missing required fields: image1, image2, label" }, { status: 400 })
    }

    const gradioPayload = {
      fn_index: 3,
      data: [image1, image2, label],
    }

    let gradioRes: Response
    try {
      gradioRes = await fetch("http://127.0.0.1:7860/run/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(gradioPayload),
        signal: AbortSignal.timeout(15000),
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
    return NextResponse.json({ success: true, data: result })
  } catch (err) {
    console.error("Feedback API error:", err)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

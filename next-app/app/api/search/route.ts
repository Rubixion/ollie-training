import { NextRequest, NextResponse } from "next/server"

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { image, mode } = body

    if (!image) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 })
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
      gradioRes = await fetch("http://127.0.0.1:7860/run/predict", {
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
      const text = await gradioRes.text()
      return NextResponse.json(
        { error: `Gradio error: ${gradioRes.status}`, detail: text },
        { status: 502 }
      )
    }

    const result = await gradioRes.json()
    return NextResponse.json(result)
  } catch (err) {
    console.error("Search API error:", err)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

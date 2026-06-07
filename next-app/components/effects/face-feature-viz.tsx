'use client'
import { useEffect, useRef } from 'react'

interface FaceFeatureVizProps {
  className?: string
  size?: number
}

const FEATURES = [
  { label: 'eye_dist', x: 0.5, y: 0.37 },
  { label: 'iris_hue_L', x: 0.34, y: 0.38 },
  { label: 'iris_hue_R', x: 0.66, y: 0.38 },
  { label: 'nose_w', x: 0.5, y: 0.54 },
  { label: 'lip_ratio', x: 0.5, y: 0.66 },
  { label: 'jaw_w', x: 0.5, y: 0.78 },
  { label: 'skin_L', x: 0.28, y: 0.54 },
  { label: 'face_ratio', x: 0.72, y: 0.54 },
  { label: 'forehead_L', x: 0.5, y: 0.24 },
  { label: 'hair_H', x: 0.5, y: 0.13 },
]

export function FaceFeatureViz({ className = '', size = 320 }: FaceFeatureVizProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    ctx.scale(dpr, dpr)

    const cx = size / 2
    const W = size

    function draw(c: CanvasRenderingContext2D, t: number) {
      c.clearRect(0, 0, W, size)

      const faceW = W * 0.34
      const faceH = size * 0.52
      const faceY = size * 0.5

      // Face silhouette
      c.save()
      c.beginPath()
      c.ellipse(cx, faceY, faceW, faceH, 0, 0, Math.PI * 2)
      const faceGrad = c.createRadialGradient(cx, faceY, faceW * 0.1, cx, faceY, faceW)
      faceGrad.addColorStop(0, 'rgba(0,212,255,0.05)')
      faceGrad.addColorStop(1, 'rgba(0,212,255,0)')
      c.fillStyle = faceGrad
      c.fill()
      c.strokeStyle = 'rgba(0,212,255,0.25)'
      c.lineWidth = 1.5
      c.stroke()
      c.restore()

      // Eyes
      const eyeY = faceY - faceH * 0.22
      const eyeX_L = cx - faceW * 0.38
      const eyeX_R = cx + faceW * 0.38
      const eyeRx = faceW * 0.2
      const eyeRy = faceH * 0.065
      ;[eyeX_L, eyeX_R].forEach((ex) => {
        c.save()
        c.beginPath()
        c.ellipse(ex, eyeY, eyeRx, eyeRy, 0, 0, Math.PI * 2)
        c.strokeStyle = 'rgba(0,212,255,0.5)'
        c.lineWidth = 1.2
        c.stroke()
        c.beginPath()
        c.arc(ex, eyeY, faceW * 0.09, 0, Math.PI * 2)
        c.strokeStyle = 'rgba(0,212,255,0.8)'
        c.lineWidth = 1
        c.stroke()
        c.beginPath()
        c.arc(ex, eyeY, faceW * 0.04, 0, Math.PI * 2)
        c.fillStyle = 'rgba(0,212,255,0.4)'
        c.fill()
        c.restore()
      })

      // Nose
      const noseBottomY = faceY + faceH * 0.05
      const noseW = faceW * 0.22
      c.save()
      c.beginPath()
      c.moveTo(cx - faceW * 0.04, eyeY + faceH * 0.12)
      c.lineTo(cx - noseW / 2, noseBottomY)
      c.lineTo(cx + noseW / 2, noseBottomY)
      c.lineTo(cx + faceW * 0.04, eyeY + faceH * 0.12)
      c.strokeStyle = 'rgba(0,212,255,0.3)'
      c.lineWidth = 1
      c.stroke()
      c.restore()

      // Mouth
      const mouthY = faceY + faceH * 0.22
      c.save()
      c.beginPath()
      c.ellipse(cx, mouthY, faceW * 0.45, faceH * 0.04, 0, 0, Math.PI)
      c.strokeStyle = 'rgba(0,212,255,0.35)'
      c.lineWidth = 1
      c.stroke()
      c.restore()

      // Landmark dots
      FEATURES.forEach((f, i) => {
        const px = f.x * W
        const py = f.y * size
        const phase = (t * 0.0015 + i * 0.7) % (Math.PI * 2)
        const pulse = 0.6 + 0.4 * Math.sin(phase)

        const ringR = 6 + 4 * Math.sin(phase)
        c.save()
        c.beginPath()
        c.arc(px, py, ringR, 0, Math.PI * 2)
        c.strokeStyle = `rgba(0,212,255,${0.15 * pulse})`
        c.lineWidth = 1
        c.stroke()
        c.restore()

        c.save()
        c.beginPath()
        c.arc(px, py, 2.5, 0, Math.PI * 2)
        c.fillStyle = `rgba(0,212,255,${pulse})`
        c.fill()
        c.restore()

        if (i < 6) {
          c.save()
          c.font = '9px monospace'
          c.textAlign = (px < cx ? 'right' : 'left') as CanvasTextAlign
          c.fillStyle = `rgba(0,212,255,${0.4 + 0.3 * pulse})`
          c.fillText(f.label, px + (px < cx ? -8 : 8), py + 3)
          c.restore()
        }
      })

      // Scanning line
      const scanPhase = ((t * 0.05) % (faceH * 2))
      const clampedScan = faceY - faceH + Math.abs(scanPhase - faceH)
      const scanGrad = c.createLinearGradient(cx - faceW, clampedScan, cx + faceW, clampedScan)
      scanGrad.addColorStop(0, 'rgba(0,212,255,0)')
      scanGrad.addColorStop(0.5, 'rgba(0,212,255,0.3)')
      scanGrad.addColorStop(1, 'rgba(0,212,255,0)')
      c.save()
      c.fillStyle = scanGrad
      c.fillRect(cx - faceW, clampedScan, faceW * 2, 1.5)
      c.restore()

      // Footer label
      c.save()
      c.font = 'bold 10px monospace'
      c.textAlign = 'center'
      c.fillStyle = 'rgba(0,212,255,0.5)'
      c.fillText('32-DIM GEOMETRIC VECTOR', cx, size - 8)
      c.restore()
    }

    const context = ctx
    function loop(ts: number) {
      draw(context, ts)
      animRef.current = requestAnimationFrame(loop)
    }

    animRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(animRef.current)
  }, [size])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size }}
      className={className}
    />
  )
}

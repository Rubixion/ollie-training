'use client'
import { useEffect, useRef } from "react"

interface Vector2D { x: number; y: number }

class Particle {
  pos: Vector2D = { x: 0, y: 0 }; vel: Vector2D = { x: 0, y: 0 }; acc: Vector2D = { x: 0, y: 0 }; target: Vector2D = { x: 0, y: 0 }
  closeEnoughTarget = 100; maxSpeed = 1.0; maxForce = 0.1; particleSize = 10; isKilled = false
  startColor = { r: 0, g: 0, b: 0 }; targetColor = { r: 0, g: 0, b: 0 }; colorWeight = 0; colorBlendRate = 0.01
  move() {
    let proximityMult = 1
    const distance = Math.sqrt(Math.pow(this.pos.x - this.target.x, 2) + Math.pow(this.pos.y - this.target.y, 2))
    if (distance < this.closeEnoughTarget) proximityMult = distance / this.closeEnoughTarget
    const towardsTarget = { x: this.target.x - this.pos.x, y: this.target.y - this.pos.y }
    const magnitude = Math.sqrt(towardsTarget.x * towardsTarget.x + towardsTarget.y * towardsTarget.y)
    if (magnitude > 0) { towardsTarget.x = (towardsTarget.x / magnitude) * this.maxSpeed * proximityMult; towardsTarget.y = (towardsTarget.y / magnitude) * this.maxSpeed * proximityMult }
    const steer = { x: towardsTarget.x - this.vel.x, y: towardsTarget.y - this.vel.y }
    const steerMag = Math.sqrt(steer.x * steer.x + steer.y * steer.y)
    if (steerMag > 0) { steer.x = (steer.x / steerMag) * this.maxForce; steer.y = (steer.y / steerMag) * this.maxForce }
    this.acc.x += steer.x; this.acc.y += steer.y
    this.vel.x += this.acc.x; this.vel.y += this.acc.y
    this.pos.x += this.vel.x; this.pos.y += this.vel.y
    this.acc.x = 0; this.acc.y = 0
  }
  draw(ctx: CanvasRenderingContext2D, drawAsPoints: boolean) {
    if (this.colorWeight < 1.0) this.colorWeight = Math.min(this.colorWeight + this.colorBlendRate, 1.0)
    const r = Math.round(this.startColor.r + (this.targetColor.r - this.startColor.r) * this.colorWeight)
    const g = Math.round(this.startColor.g + (this.targetColor.g - this.startColor.g) * this.colorWeight)
    const b = Math.round(this.startColor.b + (this.targetColor.b - this.startColor.b) * this.colorWeight)
    ctx.fillStyle = `rgb(${r},${g},${b})`
    if (drawAsPoints) { ctx.fillRect(this.pos.x, this.pos.y, 2, 2) }
    else { ctx.beginPath(); ctx.arc(this.pos.x, this.pos.y, this.particleSize / 2, 0, Math.PI * 2); ctx.fill() }
  }
  kill(width: number, height: number) {
    if (!this.isKilled) {
      const randomX = Math.random() * width; const randomY = Math.random() * height
      const mag = (width + height) / 2
      const dir = { x: randomX - width / 2, y: randomY - height / 2 }
      const len = Math.sqrt(dir.x * dir.x + dir.y * dir.y)
      if (len > 0) { dir.x = (dir.x / len) * mag; dir.y = (dir.y / len) * mag }
      this.target = { x: width / 2 + dir.x, y: height / 2 + dir.y }
      this.startColor = { r: this.startColor.r + (this.targetColor.r - this.startColor.r) * this.colorWeight, g: this.startColor.g + (this.targetColor.g - this.startColor.g) * this.colorWeight, b: this.startColor.b + (this.targetColor.b - this.startColor.b) * this.colorWeight }
      this.targetColor = { r: 0, g: 0, b: 0 }; this.colorWeight = 0; this.isKilled = true
    }
  }
}

interface ParticleTextEffectProps { words?: string[] }

export function ParticleTextEffect({ words = ["OLLIE"] }: ParticleTextEffectProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number | undefined>(undefined)
  const particlesRef = useRef<Particle[]>([])
  const frameCountRef = useRef(0)
  const wordIndexRef = useRef(0)
  const mouseRef = useRef({ x: 0, y: 0, isPressed: false, isRightClick: false })
  const pixelSteps = 6

  const nextWord = (word: string, canvas: HTMLCanvasElement) => {
    const offscreen = document.createElement("canvas")
    offscreen.width = canvas.width; offscreen.height = canvas.height
    const offCtx = offscreen.getContext("2d")!
    offCtx.fillStyle = "white"
    offCtx.font = `bold ${Math.floor(canvas.height * 0.52)}px Arial`
    offCtx.textAlign = "center"; offCtx.textBaseline = "middle"
    offCtx.fillText(word, canvas.width / 2, canvas.height / 2)
    const imageData = offCtx.getImageData(0, 0, canvas.width, canvas.height); const pixels = imageData.data
    const newColor = { r: 100, g: 130, b: 210 }
    const particles = particlesRef.current; let particleIndex = 0
    const coordsIndexes: number[] = []
    for (let i = 0; i < pixels.length; i += pixelSteps * 4) coordsIndexes.push(i)
    for (let i = coordsIndexes.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1));[coordsIndexes[i], coordsIndexes[j]] = [coordsIndexes[j], coordsIndexes[i]] }
    for (const coordIndex of coordsIndexes) {
      if (pixels[coordIndex + 3] > 0) {
        const x = (coordIndex / 4) % canvas.width; const y = Math.floor(coordIndex / 4 / canvas.width)
        let particle: Particle
        if (particleIndex < particles.length) { particle = particles[particleIndex]; particle.isKilled = false; particleIndex++ }
        else {
          particle = new Particle()
          const mag = (canvas.width + canvas.height) / 2
          const rx = Math.random() * canvas.width; const ry = Math.random() * canvas.height
          const dir = { x: rx - canvas.width / 2, y: ry - canvas.height / 2 }
          const len = Math.sqrt(dir.x * dir.x + dir.y * dir.y)
          if (len > 0) { dir.x = (dir.x / len) * mag; dir.y = (dir.y / len) * mag }
          particle.pos = { x: canvas.width / 2 + dir.x, y: canvas.height / 2 + dir.y }
          particle.maxSpeed = Math.random() * 6 + 4; particle.maxForce = particle.maxSpeed * 0.05
          particle.particleSize = Math.random() * 6 + 6; particle.colorBlendRate = Math.random() * 0.0275 + 0.0025
          particles.push(particle)
        }
        particle.startColor = { r: particle.startColor.r + (particle.targetColor.r - particle.startColor.r) * particle.colorWeight, g: particle.startColor.g + (particle.targetColor.g - particle.startColor.g) * particle.colorWeight, b: particle.startColor.b + (particle.targetColor.b - particle.startColor.b) * particle.colorWeight }
        particle.targetColor = newColor; particle.colorWeight = 0; particle.target = { x, y }
      }
    }
    for (let i = particleIndex; i < particles.length; i++) particles[i].kill(canvas.width, canvas.height)
  }

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    canvas.width = 2000; canvas.height = 300
    nextWord(words[0], canvas)
    const animate = () => {
      const ctx = canvas.getContext("2d")!; const particles = particlesRef.current
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i]; p.move(); p.draw(ctx, true)
        if (p.isKilled && (p.pos.x < 0 || p.pos.x > canvas.width || p.pos.y < 0 || p.pos.y > canvas.height)) particles.splice(i, 1)
      }
      frameCountRef.current++
      if (frameCountRef.current % 240 === 0) { wordIndexRef.current = (wordIndexRef.current + 1) % words.length; nextWord(words[wordIndexRef.current], canvas) }
      animRef.current = requestAnimationFrame(animate)
    }
    animate()
    const onDown = (e: MouseEvent) => { mouseRef.current.isPressed = true; mouseRef.current.isRightClick = e.button === 2; const r = canvas.getBoundingClientRect(); mouseRef.current.x = e.clientX - r.left; mouseRef.current.y = e.clientY - r.top }
    const onUp = () => { mouseRef.current.isPressed = false; mouseRef.current.isRightClick = false }
    const onMove = (e: MouseEvent) => { const r = canvas.getBoundingClientRect(); mouseRef.current.x = e.clientX - r.left; mouseRef.current.y = e.clientY - r.top }
    canvas.addEventListener("mousedown", onDown); canvas.addEventListener("mouseup", onUp); canvas.addEventListener("mousemove", onMove); canvas.addEventListener("contextmenu", e => e.preventDefault())
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); canvas.removeEventListener("mousedown", onDown); canvas.removeEventListener("mouseup", onUp); canvas.removeEventListener("mousemove", onMove) }
  }, [words])

  return <canvas ref={canvasRef} style={{ width: "100%", height: "auto", display: "block" }} />
}

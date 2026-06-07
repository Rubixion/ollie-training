"use client"

import { useEffect, useRef, useState } from "react"
import { motion } from "framer-motion"
import type {
  WebGLRenderer,
  Scene,
  PerspectiveCamera,
  Mesh,
  Points,
  BufferAttribute,
  MeshBasicMaterial,
  Vector3,
} from "three"

// CSS fallback neural network diagram
function CssFallback() {
  const layers = [
    { label: "Input", nodes: 1, size: "large" },
    { label: "Conv 1", nodes: 4, size: "small" },
    { label: "Conv 2", nodes: 4, size: "small" },
    { label: "Features", nodes: 8, size: "tiny" },
    { label: "Embed", nodes: 4, size: "small" },
    { label: "Compare", nodes: 2, size: "small" },
  ]

  return (
    <div className="w-full flex items-center justify-center gap-3 md:gap-6 py-12 px-4 overflow-x-auto">
      {layers.map((layer, li) => (
        <div key={li} className="flex flex-col items-center gap-2 shrink-0">
          <div className="flex flex-col items-center gap-1.5">
            {Array.from({ length: layer.nodes }).map((_, ni) => (
              <motion.div
                key={ni}
                animate={{ opacity: [0.4, 1, 0.4], scale: [0.95, 1.05, 0.95] }}
                transition={{
                  repeat: Infinity,
                  duration: 2,
                  delay: (li * 0.3 + ni * 0.1) % 2,
                  ease: "easeInOut",
                }}
                className={`rounded-full border border-[--ollie-cyan]/60 bg-[--ollie-cyan]/10 shadow-[0_0_8px_var(--ollie-glow)]
                  ${layer.size === "large" ? "w-8 h-8" : layer.size === "small" ? "w-5 h-5" : "w-3 h-3"}`}
              />
            ))}
          </div>
          <span className="text-white/30 text-[9px] font-mono uppercase tracking-wide">{layer.label}</span>
        </div>
      ))}
    </div>
  )
}

async function buildThreeScene(container: HTMLDivElement): Promise<() => void> {
  const THREE = await import("three")

  const width = container.clientWidth || 800
  const height = container.clientHeight || 400

  const renderer: WebGLRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setSize(width, height)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setClearColor(0x000000, 0)
  container.appendChild(renderer.domElement)

  const scene: Scene = new THREE.Scene()
  const camera: PerspectiveCamera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100)
  camera.position.set(0, 0, 7)

  const cyanColor = new THREE.Color(0x49d9c8)
  const purpleColor = new THREE.Color(0x9b6ef3)

  // Layers: [x position, nodeCount, nodeSize]
  const layerDefs: [number, number, number][] = [
    [-4.5, 1, 0.35],
    [-2.8, 4, 0.18],
    [-1.2, 4, 0.18],
    [0.5, 8, 0.11],
    [2.2, 4, 0.18],
    [3.8, 2, 0.22],
  ]

  const nodePositions: Vector3[][] = []
  const nodeMeshes: Mesh[] = []

  layerDefs.forEach(([x, count, size], li) => {
    const positions: Vector3[] = []
    const totalHeight = (count - 1) * 0.55
    for (let i = 0; i < count; i++) {
      const y = -totalHeight / 2 + i * 0.55
      const pos = new THREE.Vector3(x, y, 0)
      positions.push(pos)

      const geo = new THREE.SphereGeometry(size, 16, 16)
      const mat: MeshBasicMaterial = new THREE.MeshBasicMaterial({
        color: li === layerDefs.length - 1 ? purpleColor : cyanColor,
        transparent: true,
        opacity: 0.85,
      })
      const mesh: Mesh = new THREE.Mesh(geo, mat)
      mesh.position.copy(pos)
      scene.add(mesh)
      nodeMeshes.push(mesh)

      const haloGeo = new THREE.SphereGeometry(size * 1.8, 16, 16)
      const haloMat: MeshBasicMaterial = new THREE.MeshBasicMaterial({
        color: li < layerDefs.length - 1 ? cyanColor : purpleColor,
        transparent: true,
        opacity: 0.07,
      })
      const halo: Mesh = new THREE.Mesh(haloGeo, haloMat)
      halo.position.copy(pos)
      scene.add(halo)
    }
    nodePositions.push(positions)
  })

  const lineMat = new THREE.LineBasicMaterial({
    color: cyanColor,
    transparent: true,
    opacity: 0.08,
  })

  for (let li = 0; li < nodePositions.length - 1; li++) {
    const from = nodePositions[li]
    const to = nodePositions[li + 1]
    for (const fPos of from) {
      for (const tPos of to) {
        const geo = new THREE.BufferGeometry().setFromPoints([fPos, tPos])
        const line = new THREE.Line(geo, lineMat)
        scene.add(line)
      }
    }
  }

  const particleCount = 30
  const particleGeo = new THREE.BufferGeometry()
  const particlePositions = new Float32Array(particleCount * 3)
  const particleData: {
    layerIdx: number
    fromNode: number
    toNode: number
    t: number
    speed: number
  }[] = []

  for (let i = 0; i < particleCount; i++) {
    const layerIdx = Math.floor(Math.random() * (nodePositions.length - 1))
    const fromNode = Math.floor(Math.random() * nodePositions[layerIdx].length)
    const toNode = Math.floor(Math.random() * nodePositions[layerIdx + 1].length)
    particleData.push({ layerIdx, fromNode, toNode, t: Math.random(), speed: 0.004 + Math.random() * 0.008 })
  }

  particleGeo.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3))
  const particleMat = new THREE.PointsMaterial({ color: cyanColor, size: 0.06, transparent: true, opacity: 0.9 })
  const particles: Points = new THREE.Points(particleGeo, particleMat)
  scene.add(particles)

  const mouseDelta = { x: 0, y: 0 }
  const cameraAngle = { x: 0, y: 0 }

  function onMouseMove(e: MouseEvent) {
    const rect = container.getBoundingClientRect()
    mouseDelta.x = ((e.clientX - rect.left) / rect.width - 0.5) * 2
    mouseDelta.y = ((e.clientY - rect.top) / rect.height - 0.5) * 2
  }
  container.addEventListener("mousemove", onMouseMove)

  function onResize() {
    const w = container.clientWidth
    const h = container.clientHeight
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h)
  }
  window.addEventListener("resize", onResize)

  let frame = 0
  let animId: number

  function animate() {
    animId = requestAnimationFrame(animate)
    frame++

    nodeMeshes.forEach((mesh, i) => {
      const mat = mesh.material as MeshBasicMaterial
      mat.opacity = 0.6 + 0.4 * Math.sin(frame * 0.03 + i * 0.5)
    })

    const posAttr = particleGeo.getAttribute("position") as BufferAttribute
    particleData.forEach((p, i) => {
      p.t += p.speed
      if (p.t > 1) {
        p.t = 0
        p.layerIdx = Math.floor(Math.random() * (nodePositions.length - 1))
        p.fromNode = Math.floor(Math.random() * nodePositions[p.layerIdx].length)
        p.toNode = Math.floor(Math.random() * nodePositions[p.layerIdx + 1].length)
      }
      const from = nodePositions[p.layerIdx][p.fromNode]
      const to = nodePositions[p.layerIdx + 1][p.toNode]
      posAttr.setXYZ(i, from.x + (to.x - from.x) * p.t, from.y + (to.y - from.y) * p.t, 0)
    })
    posAttr.needsUpdate = true

    cameraAngle.x += (mouseDelta.x * 0.4 - cameraAngle.x) * 0.05
    cameraAngle.y += (-mouseDelta.y * 0.25 - cameraAngle.y) * 0.05
    camera.position.x = Math.sin(cameraAngle.x) * 7
    camera.position.y = cameraAngle.y * 2
    camera.position.z = Math.cos(cameraAngle.x) * 7
    camera.lookAt(0, 0, 0)

    renderer.render(scene, camera)
  }

  animate()

  return () => {
    container.removeEventListener("mousemove", onMouseMove)
    window.removeEventListener("resize", onResize)
    cancelAnimationFrame(animId)
    renderer.dispose()
    if (container.contains(renderer.domElement)) {
      container.removeChild(renderer.domElement)
    }
  }
}

function ThreeViz() {
  const canvasRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = canvasRef.current
    if (!container) return

    let cleanup: (() => void) | undefined

    buildThreeScene(container)
      .then((fn) => { cleanup = fn })
      .catch((err) => console.warn("Three.js init failed:", err))

    return () => { cleanup?.() }
  }, [])

  return <div ref={canvasRef} className="w-full h-full" />
}

export function NeuralViz() {
  const [webGLAvailable, setWebGLAvailable] = useState(true)

  useEffect(() => {
    try {
      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("webgl") || canvas.getContext("experimental-webgl")
      if (!ctx) setWebGLAvailable(false)
    } catch {
      setWebGLAvailable(false)
    }
  }, [])

  const layerLabels = [
    { label: "Input Face", sub: "224×224 px" },
    { label: "Conv Layers", sub: "Feature maps" },
    { label: "Dense Layer", sub: "512-dim" },
    { label: "Embedding", sub: "128-dim" },
    { label: "Distance", sub: "Similarity score" },
  ]

  return (
    <section id="how-it-works" className="py-24 md:py-32 px-6 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="inline-block text-xs font-semibold tracking-widest text-[--ollie-cyan] uppercase mb-3">
            Architecture
          </span>
          <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight">
            How Ollie Thinks
          </h2>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.97 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="relative rounded-2xl border border-white/5 bg-white/[0.02] overflow-hidden"
          style={{ height: 360 }}
        >
          {webGLAvailable ? <ThreeViz /> : <CssFallback />}

          {/* Layer labels */}
          <div className="absolute bottom-4 left-0 right-0 flex justify-around px-4 pointer-events-none">
            {layerLabels.map((l, i) => (
              <div key={i} className="text-center">
                <p className="text-white/50 text-[10px] font-semibold">{l.label}</p>
                <p className="text-white/20 text-[9px]">{l.sub}</p>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mt-10 grid md:grid-cols-3 gap-6"
        >
          {[
            {
              step: "01",
              title: "Two images enter",
              desc: "Your uploaded photo and each celebrity image are fed simultaneously into twin CNN branches.",
            },
            {
              step: "02",
              title: "Identical CNN layers",
              desc: "Both images pass through the same convolutional layers — weights are shared. Each layer learns increasingly abstract facial features.",
            },
            {
              step: "03",
              title: "Embedding distance",
              desc: "The distance between their 128-dimensional embeddings determines similarity. Smaller distance = stronger match.",
            },
          ].map((item) => (
            <div key={item.step} className="p-5 rounded-xl bg-white/[0.02] border border-white/5">
              <span className="text-[--ollie-cyan]/50 text-xs font-mono">{item.step}</span>
              <h3 className="text-white font-bold mt-2 mb-2">{item.title}</h3>
              <p className="text-white/40 text-sm leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}

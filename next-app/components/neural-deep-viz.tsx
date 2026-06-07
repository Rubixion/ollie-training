"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { X, ArrowDown } from "lucide-react"

interface LayerDef {
  x: number
  count: number
  label: string
  simple: string
  detail: string
  sectionId?: string
  isInput?: boolean
  isDistance?: boolean
}

const TOP_LAYERS: LayerDef[] = [
  {
    x: -6, count: 1, label: "Your photo",
    simple: "This is your photo - just a grid of colour numbers that the AI reads one small region at a time.",
    detail: "96x96x3 RGB tensor, pixel values normalised to [-1,1]. Input shape: [1, 3, 96, 96]. Each pixel is a float32 triple.",
    sectionId: "s-backbone", isInput: true,
  },
  {
    x: -4, count: 4, label: "Edge detector",
    simple: "The first layer slides a tiny window across your face picking up edges, corners, and colour changes.",
    detail: "Conv2d(3→64, kernel=3, stride=1, padding=1) + BatchNorm2d + ReLU. Detects low-level gradients. Output: [1, 64, 96, 96].",
    sectionId: "s-backbone",
  },
  {
    x: -2, count: 4, label: "Shape finder",
    simple: "A deeper layer that starts assembling edges into shapes - eye corners, nose bridges, jaw lines.",
    detail: "ResBlock(64→128, stride=2) with 1x1 projection shortcut. Skip connection prevents vanishing gradients. Output: [1, 128, 48, 48].",
    sectionId: "s-backbone",
  },
  {
    x: 0, count: 6, label: "Feature mapper",
    simple: "By here the network notices whole facial regions - eye spacing, nose width, forehead height.",
    detail: "ResBlock(128→256, stride=2). Receptive field ~40px of original image. Mid-level facial structure. Output: [1, 256, 24, 24].",
    sectionId: "s-backbone",
  },
  {
    x: 2, count: 6, label: "Identity layer",
    simple: "The deepest layer - it now picks up high-level traits unique to one person, not just any face.",
    detail: "ResBlock(256→512, stride=2). Receptive field ~80px. High-level identity features emerge. Output: [1, 512, 12, 12].",
    sectionId: "s-backbone",
  },
  {
    x: 4, count: 3, label: "Summariser",
    simple: "Compresses everything into a single flat summary - like turning a detailed report into one line.",
    detail: "Adaptive average pool 12x12 → 1x1. Spatial translation invariance. Output: [1, 512]. No learned parameters.",
    sectionId: "s-backbone",
  },
  {
    x: 6, count: 2, label: "Face fingerprint",
    simple: "Your face is now 256 numbers. This is your unique fingerprint - similar faces land close together.",
    detail: "Linear(512→256) + L2 normalisation. Projects onto unit hypersphere in R^256. ||e||_2 = 1.0 by construction. Cosine distance is equivalent to half squared L2.",
    sectionId: "s-siamese",
  },
]

const BOTTOM_LAYERS: LayerDef[] = [
  {
    x: -6, count: 1, label: "Celebrity photo",
    simple: "A celebrity photo runs through an identical copy of the same network at the same time.",
    detail: "Second face - Siamese twin input. Same preprocessing pipeline. All weights W and biases b are shared with the top network.",
    sectionId: "s-siamese", isInput: true,
  },
  {
    x: -4, count: 4, label: "Edge detector",
    simple: "Same first step - scanning for edges using the same patterns the network already learned.",
    detail: "Exact same Conv2d weights as the top network. Weight sharing is the defining property of a Siamese architecture.",
    sectionId: "s-siamese",
  },
  {
    x: -2, count: 4, label: "Shape finder",
    simple: "Same layer, same learned rules - both faces are measured using identical criteria.",
    detail: "Same ResBlock(64→128) weights. Ensures both embeddings live in the same geometric space. Any transformation applied to face 1 applies equally to face 2.",
    sectionId: "s-siamese",
  },
  {
    x: 0, count: 6, label: "Feature mapper",
    simple: "The second face's regions are mapped out using the exact same rules as the first face.",
    detail: "Same ResBlock(128→256) weights. No separate learning path for input 2. Symmetry enforced by architecture, not training.",
    sectionId: "s-siamese",
  },
  {
    x: 2, count: 6, label: "Identity layer",
    simple: "High-level identity features of the celebrity face come out here, measured identically to yours.",
    detail: "Same ResBlock(256→512) weights. Both faces processed through identical computational graph. Shared gradient flow during training.",
    sectionId: "s-siamese",
  },
  {
    x: 4, count: 3, label: "Summariser",
    simple: "Same compression step - the celebrity face becomes a single summary ready to compare with yours.",
    detail: "Same adaptive average pool 12x12 → 1x1. Output vectors are directly comparable because the same transformation was applied.",
    sectionId: "s-siamese",
  },
  {
    x: 6, count: 2, label: "Face fingerprint",
    simple: "The celebrity gets their 256-number fingerprint. Now both fingerprints are in the same space.",
    detail: "Same Linear(512→256) + L2 normalisation. Distance between unit vectors is geometrically meaningful - cos(e1,e2) = 1 - d^2/2.",
    sectionId: "s-siamese",
  },
]

const DISTANCE_LAYER: LayerDef = {
  x: 8.5, count: 1, label: "Similarity score",
  simple: "The gap between the two fingerprints - small gap means the same person, large gap means different people.",
  detail: "L2 distance ||f(x1) - f(x2)||_2, range [0,2] on unit sphere. Threshold ~0.5. BCE head: sigmoid(|e1-e2| * W + b) -> similarity in [0,1].",
  sectionId: "s-contrastive",
  isDistance: true,
}

function nodeYPositions(count: number, spacing = 0.5): number[] {
  const total = (count - 1) * spacing
  return Array.from({ length: count }, (_, i) => i * spacing - total / 2)
}

interface HoveredNode {
  label: string
  simple: string
  detail: string
  sectionId?: string
  screenX: number
  screenY: number
}

export function NeuralDeepViz() {
  const canvasRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HoveredNode | null>(null)
  const [tooltip, setTooltip] = useState<HoveredNode | null>(null)
  const [devModal, setDevModal] = useState<{ label: string; detail: string } | null>(null)
  const rafRef = useRef<number | undefined>(undefined)
  const sceneDataRef = useRef<{
    renderer: THREE.WebGLRenderer
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    controls: OrbitControls
    nodeMeshes: Array<{ mesh: THREE.Mesh; layer: LayerDef }>
    signalParticles: THREE.Mesh[]
    signalProgress: number[]
    signalPaths: Array<{ start: THREE.Vector3; end: THREE.Vector3 }>
    signalWobble: Array<{ px: number; py: number; amp: number; freq: number; phase: number }>
    raycaster: THREE.Raycaster
    mouse: THREE.Vector2
  } | null>(null)

  const buildScene = useCallback(() => {
    if (!canvasRef.current) return
    const W = canvasRef.current.clientWidth
    const H = canvasRef.current.clientHeight

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(W, H)
    renderer.setClearColor(0x000000, 0)
    canvasRef.current.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.fog = new THREE.Fog(0x000000, 20, 60)

    const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 200)
    camera.position.set(0, 2, 16)
    camera.lookAt(0, -0.5, 0)

    scene.add(new THREE.AmbientLight(0xffffff, 0.3))
    const topLight = new THREE.PointLight(0x6482D2, 4, 28)
    topLight.position.set(-3, 6, 6)
    scene.add(topLight)
    const bottomLight = new THREE.PointLight(0xA07EC8, 4, 28)
    bottomLight.position.set(-3, -6, 6)
    scene.add(bottomLight)
    const distLight = new THREE.PointLight(0xD4A050, 5, 18)
    distLight.position.set(9, 0, 4)
    scene.add(distLight)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 5
    controls.maxDistance = 40
    controls.target.set(1, -1.5, 0)
    controls.update()

    const nodeMeshes: Array<{ mesh: THREE.Mesh; layer: LayerDef }> = []

    function buildNetworkRow(layers: LayerDef[], yOffset: number, nodeColor: number, emissiveColor: number) {
      for (const layer of layers) {
        const yPositions = nodeYPositions(layer.count, 0.55)
        for (const yLocal of yPositions) {
          const radius = layer.isInput ? 0.38 : layer.isDistance ? 0.42 : 0.14
          const geo = new THREE.SphereGeometry(radius, 16, 16)
          const mat = new THREE.MeshStandardMaterial({
            color: nodeColor,
            emissive: emissiveColor,
            emissiveIntensity: layer.isInput ? 1.2 : layer.isDistance ? 1.4 : 0.6,
            roughness: 0.9,
            metalness: 0.0,
            transparent: true,
            opacity: layer.isInput ? 0.85 : layer.isDistance ? 0.85 : 0.75,
          })
          const mesh = new THREE.Mesh(geo, mat)
          mesh.position.set(layer.x, yOffset + yLocal, 0)
          scene.add(mesh)
          nodeMeshes.push({ mesh, layer })
        }
      }
    }

    buildNetworkRow(TOP_LAYERS, 1.75, 0x6482D2, 0x2A3A72)
    buildNetworkRow(BOTTOM_LAYERS, -1.75, 0xA07EC8, 0x4A2870)
    buildNetworkRow([DISTANCE_LAYER], 0, 0xD4A050, 0x705020)

    function buildConnections(layers: LayerDef[], yOffset: number, lineColor: number) {
      for (let li = 0; li < layers.length - 1; li++) {
        const fromLayer = layers[li]
        const toLayer = layers[li + 1]
        const fromYs = nodeYPositions(fromLayer.count, 0.55)
        const toYs = nodeYPositions(toLayer.count, 0.55)
        const points: THREE.Vector3[] = []
        for (const fy of fromYs) {
          for (const ty of toYs) {
            points.push(
              new THREE.Vector3(fromLayer.x, yOffset + fy, 0),
              new THREE.Vector3(toLayer.x, yOffset + ty, 0)
            )
          }
        }
        const geo = new THREE.BufferGeometry().setFromPoints(points)
        const mat = new THREE.LineBasicMaterial({ color: lineColor, transparent: true, opacity: 0.10 })
        scene.add(new THREE.LineSegments(geo, mat))
      }
    }

    buildConnections(TOP_LAYERS, 1.75, 0x6482D2)
    buildConnections(BOTTOM_LAYERS, -1.75, 0xA07EC8)

    function connectToDistance(lastLayer: LayerDef, yOffset: number) {
      const fromYs = nodeYPositions(lastLayer.count, 0.55)
      const points: THREE.Vector3[] = []
      for (const fy of fromYs) {
        points.push(
          new THREE.Vector3(lastLayer.x, yOffset + fy, 0),
          new THREE.Vector3(DISTANCE_LAYER.x, 0, 0)
        )
      }
      const geo = new THREE.BufferGeometry().setFromPoints(points)
      const mat = new THREE.LineBasicMaterial({ color: 0xD4A050, transparent: true, opacity: 0.18 })
      scene.add(new THREE.LineSegments(geo, mat))
    }

    connectToDistance(TOP_LAYERS[TOP_LAYERS.length - 1], 1.75)
    connectToDistance(BOTTOM_LAYERS[BOTTOM_LAYERS.length - 1], -1.75)

    function buildSignalPaths(layers: LayerDef[], yOffset: number) {
      const paths: Array<{ start: THREE.Vector3; end: THREE.Vector3 }> = []
      for (let li = 0; li < layers.length - 1; li++) {
        paths.push({
          start: new THREE.Vector3(layers[li].x, yOffset, 0),
          end: new THREE.Vector3(layers[li + 1].x, yOffset, 0),
        })
      }
      return paths
    }

    const topPaths = buildSignalPaths(TOP_LAYERS, 1.75)
    const bottomPaths = buildSignalPaths(BOTTOM_LAYERS, -1.75)
    const distPathTop = {
      start: new THREE.Vector3(TOP_LAYERS[TOP_LAYERS.length - 1].x, 1.75, 0),
      end: new THREE.Vector3(DISTANCE_LAYER.x, 0, 0),
    }
    const distPathBottom = {
      start: new THREE.Vector3(BOTTOM_LAYERS[BOTTOM_LAYERS.length - 1].x, -1.75, 0),
      end: new THREE.Vector3(DISTANCE_LAYER.x, 0, 0),
    }
    const signalPaths = [...topPaths, ...bottomPaths, distPathTop, distPathBottom]

    const signalParticles: THREE.Mesh[] = []
    const signalProgress: number[] = []
    const signalWobble: Array<{ px: number; py: number; amp: number; freq: number; phase: number }> = []

    const topCount = topPaths.length
    const bottomCount = bottomPaths.length

    for (let i = 0; i < signalPaths.length; i++) {
      const isTop = i < topCount
      const isBottom = i >= topCount && i < topCount + bottomCount
      const isDist = i >= topCount + bottomCount
      const partColor = isTop ? 0x8AAAE8 : isBottom ? 0xC09AE8 : 0xE8C070
      const partEmissive = isTop ? 0x6482D2 : isBottom ? 0xA07EC8 : 0xD4A050

      const geo = new THREE.SphereGeometry(0.07, 8, 8)
      const mat = new THREE.MeshStandardMaterial({
        color: partColor,
        emissive: partEmissive,
        emissiveIntensity: 1.8,
        transparent: true,
        opacity: 0,
        stencilWrite: false,
        stencilRef: 0,
        stencilFunc: THREE.EqualStencilFunc,
      })
      const mesh = new THREE.Mesh(geo, mat)
      scene.add(mesh)
      signalParticles.push(mesh)

      // Stagger start: top/bottom pairs share the same initial phase so they're symmetrical
      if (isDist) {
        signalProgress.push(0.5)
      } else if (isTop) {
        signalProgress.push(0.2 + (i / topCount) * 0.6)
      } else {
        signalProgress.push(0.2 + ((i - topCount) / bottomCount) * 0.6)
      }

      const path = signalPaths[i]
      const dir = new THREE.Vector3().subVectors(path.end, path.start).normalize()

      if (isBottom) {
        // Mirror the corresponding top path's wobble for vertical symmetry
        const top = signalWobble[i - topCount]
        signalWobble.push({ px: top.px, py: -top.py, amp: top.amp, freq: top.freq, phase: top.phase })
      } else {
        signalWobble.push({
          px: -dir.y,
          py: dir.x,
          amp: isDist ? 0 : 0.22 + Math.random() * 0.18,
          freq: isDist ? 1 : 2 + Math.random() * 2,
          phase: Math.random() * Math.PI * 2,
        })
      }
    }

    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()

    sceneDataRef.current = { renderer, scene, camera, controls, nodeMeshes, signalParticles, signalProgress, signalPaths, signalWobble, raycaster, mouse }
  }, [])

  const animate = useCallback(() => {
    const data = sceneDataRef.current
    if (!data) return
    data.controls.update()

    for (let i = 0; i < data.signalParticles.length; i++) {
      // Loop only within [0.2, 0.8] - at both endpoints opacity=0 so reset is seamless
      data.signalProgress[i] += 0.003
      if (data.signalProgress[i] > 0.8) data.signalProgress[i] = 0.2

      const t = data.signalProgress[i]
      const w = data.signalWobble[i]
      const mesh = data.signalParticles[i]
      const mat = mesh.material as THREE.MeshStandardMaterial
      const path = data.signalPaths[i]

      const tNorm = (t - 0.2) / 0.6            // remap [0.2, 0.8] → [0, 1]
      const raw = Math.sin(tNorm * Math.PI)
      const envelope = raw * raw * raw
      mat.opacity = envelope * 0.95

      const wobble = Math.sin(t * Math.PI * w.freq + w.phase) * w.amp * envelope
      const base = new THREE.Vector3().lerpVectors(path.start, path.end, t)
      mesh.position.set(base.x + w.px * wobble, base.y + w.py * wobble, base.z)
    }

    data.raycaster.setFromCamera(data.mouse, data.camera)
    const intersects = data.raycaster.intersectObjects(data.nodeMeshes.map((n) => n.mesh))

    if (intersects.length > 0) {
      const hit = intersects[0].object as THREE.Mesh
      const nodeData = data.nodeMeshes.find((n) => n.mesh === hit)
      if (nodeData) {
        const pos = nodeData.mesh.position.clone().project(data.camera)
        const canvas = canvasRef.current
        if (canvas) {
          const rect = canvas.getBoundingClientRect()
          const screenX = ((pos.x + 1) / 2) * rect.width
          const screenY = ((-pos.y + 1) / 2) * rect.height
          const next: HoveredNode = {
            label: nodeData.layer.label,
            simple: nodeData.layer.simple,
            detail: nodeData.layer.detail,
            sectionId: nodeData.layer.sectionId,
            screenX,
            screenY,
          }
          if (!tooltipRef.current || tooltipRef.current.label !== next.label) {
            tooltipRef.current = next
            setTooltip(next)
          }
        }
      }
    } else {
      if (tooltipRef.current) {
        tooltipRef.current = null
        setTooltip(null)
      }
    }

    data.renderer.render(data.scene, data.camera)
    rafRef.current = requestAnimationFrame(animate)
  }, [])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas || !sceneDataRef.current) return
    const rect = canvas.getBoundingClientRect()
    sceneDataRef.current.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
    sceneDataRef.current.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
  }, [])

  const handleResize = useCallback(() => {
    const data = sceneDataRef.current
    const canvas = canvasRef.current
    if (!data || !canvas) return
    data.camera.aspect = canvas.clientWidth / canvas.clientHeight
    data.camera.updateProjectionMatrix()
    data.renderer.setSize(canvas.clientWidth, canvas.clientHeight)
  }, [])

  useEffect(() => {
    buildScene()
    rafRef.current = requestAnimationFrame(animate)
    window.addEventListener("resize", handleResize)
    const canvas = canvasRef.current
    if (canvas) canvas.addEventListener("mousemove", handleMouseMove)
    return () => {
      window.removeEventListener("resize", handleResize)
      if (canvas) canvas.removeEventListener("mousemove", handleMouseMove)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      const data = sceneDataRef.current
      if (data) {
        data.controls.dispose()
        data.renderer.dispose()
        data.scene.traverse((obj) => {
          if (obj instanceof THREE.Mesh) {
            obj.geometry.dispose()
            if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose())
            else obj.material.dispose()
          }
        })
        // Dispose signal particles not currently in the scene (never added or already removed)
        for (const p of data.signalParticles) {
          if (!p.parent) {
            p.geometry.dispose()
            ;(p.material as THREE.Material).dispose()
          }
        }
        // Force-remove ALL canvas children so a stale frozen frame can't linger (StrictMode)
        if (canvas) {
          while (canvas.firstChild) canvas.removeChild(canvas.firstChild)
        }
      }
      sceneDataRef.current = null
    }
  }, [buildScene, animate, handleResize, handleMouseMove])

  return (
    <div className="relative w-full">
      {/* Canvas */}
      <div
        ref={canvasRef}
        className="w-full rounded-2xl overflow-hidden border border-white/10 bg-black/60"
        style={{ height: "70vh" }}
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute z-20 bg-[#0a0a0a] border border-white/12 rounded-2xl p-4 max-w-[260px] shadow-2xl"
          style={{
            left: Math.min(tooltip.screenX + 16, (canvasRef.current?.clientWidth ?? 800) - 280),
            top: Math.max(tooltip.screenY - 80, 10),
            pointerEvents: "auto",
          }}
        >
          <div className="text-white/40 text-[9px] font-bold tracking-widest uppercase mb-1.5">
            {tooltip.label}
          </div>
          <p className="text-white/75 text-xs leading-relaxed mb-3">
            {tooltip.simple}
          </p>
          <div className="flex gap-2">
            {tooltip.sectionId && (
              <a
                href={`#${tooltip.sectionId}`}
                className="inline-flex items-center gap-1 text-[10px] font-semibold text-white/50 hover:text-white border border-white/15 hover:border-white/30 rounded-lg px-2.5 py-1.5 transition-colors"
              >
                <ArrowDown size={10} />
                Full explanation
              </a>
            )}
            <button
              onClick={() => setDevModal({ label: tooltip.label, detail: tooltip.detail })}
              className="inline-flex items-center gap-1 text-[10px] font-semibold text-white/30 hover:text-white/60 border border-white/8 hover:border-white/20 rounded-lg px-2.5 py-1.5 transition-colors"
            >
              For the devs
            </button>
          </div>
        </div>
      )}

      {/* Controls hint */}
      <p className="text-center text-white/20 text-xs mt-4 tracking-wide">
        Drag to orbit &middot; Scroll to zoom &middot; Hover a node for details
      </p>

      {/* Dev modal */}
      {devModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-6"
          onClick={() => setDevModal(null)}
        >
          <div
            className="relative bg-[#0d0d0d] border border-white/12 rounded-2xl p-7 max-w-lg w-full shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setDevModal(null)}
              className="absolute top-4 right-4 text-white/30 hover:text-white/70 transition-colors"
            >
              <X size={16} />
            </button>
            <div className="text-white/35 text-[9px] font-bold tracking-widest uppercase mb-1">
              For the devs
            </div>
            <div className="text-white font-bold text-base mb-4">{devModal.label}</div>
            <pre className="text-white/55 text-xs font-mono leading-relaxed whitespace-pre-wrap break-words">
              {devModal.detail}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}

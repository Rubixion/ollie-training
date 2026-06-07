"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"

interface LayerDef {
  x: number
  count: number
  label: string
  detail: string
  sectionId?: string
  isInput?: boolean
  isDistance?: boolean
}

const TOP_LAYERS: LayerDef[] = [
  { x: -6, count: 1, label: "INPUT IMAGE", detail: "96Ã—96Ã—3 RGB face image. Normalized to [-1,1] range. Each pixel becomes a floating-point value.", sectionId: "s-backbone", isInput: true },
  { x: -4, count: 4, label: "CONV 3Ã—3", detail: "64 channels, 3Ã—3 convolutions with BatchNorm + ReLU. Detects low-level edges and textures.", sectionId: "s-backbone" },
  { x: -2, count: 4, label: "RESIDUAL 128ch", detail: "ResBlock: stride=2, 128 channels. Skip connection via 1Ã—1 projection prevents vanishing gradients.", sectionId: "s-backbone" },
  { x: 0, count: 6, label: "RESIDUAL 256ch", detail: "ResBlock: stride=2, 256 channels. Feature map shrinks to 24Ã—24. Learns mid-level facial structure.", sectionId: "s-backbone" },
  { x: 2, count: 6, label: "RESIDUAL 512ch", detail: "ResBlock: stride=2, 512 channels. Feature map: 12Ã—12. High-level identity features emerge here.", sectionId: "s-backbone" },
  { x: 4, count: 3, label: "GLOBAL AVG POOL", detail: "6Ã—6 adaptive average pooling collapses spatial dims â†’ 512-d vector. Spatial invariance.", sectionId: "s-backbone" },
  { x: 6, count: 2, label: "L2-NORM 256d", detail: "FC 512â†’256 + L2 normalization. Projects to unit hypersphere. Enables cosine similarity.", sectionId: "s-siamese" },
]

const BOTTOM_LAYERS: LayerDef[] = [
  { x: -6, count: 1, label: "INPUT IMAGE", detail: "Second face â€” Siamese twin input. Both networks share identical weights.", sectionId: "s-siamese", isInput: true },
  { x: -4, count: 4, label: "CONV 3Ã—3", detail: "Exact same weights as the top network. Weight sharing is what makes it 'Siamese'.", sectionId: "s-siamese" },
  { x: -2, count: 4, label: "RESIDUAL 128ch", detail: "Same weights â€” stride=2, 128 channels. Ensures both embeddings live in the same space.", sectionId: "s-siamese" },
  { x: 0, count: 6, label: "RESIDUAL 256ch", detail: "Same weights â€” stride=2, 256 channels. No separate learning for each input.", sectionId: "s-siamese" },
  { x: 2, count: 6, label: "RESIDUAL 512ch", detail: "Same weights â€” stride=2, 512 channels. Both faces processed identically.", sectionId: "s-siamese" },
  { x: 4, count: 3, label: "GLOBAL AVG POOL", detail: "Same pooling operation â€” 6Ã—6 â†’ 512d. Output vectors are comparable by design.", sectionId: "s-siamese" },
  { x: 6, count: 2, label: "L2-NORM 256d", detail: "Same FC + L2 normalization. Distance between embeddings is now meaningful.", sectionId: "s-siamese" },
]

const DISTANCE_LAYER: LayerDef = {
  x: 8, count: 1, label: "DISTANCE",
  detail: "L2 distance: â€–f(xâ‚) - f(xâ‚‚)â€–â‚‚. Near zero = same person. Near 2 = different. Threshold â‰ˆ 0.5.",
  sectionId: "s-contrastive",
  isDistance: true,
}

// Generate Y positions for nodes in a layer
function nodeYPositions(count: number, spacing = 0.5): number[] {
  const total = (count - 1) * spacing
  return Array.from({ length: count }, (_, i) => i * spacing - total / 2)
}

interface HoveredNode {
  label: string
  detail: string
  sectionId?: string
  screenX: number
  screenY: number
}

export function NeuralDeepViz() {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HoveredNode | null>(null)
  const [tooltip, setTooltip] = useState<HoveredNode | null>(null)
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
    raycaster: THREE.Raycaster
    mouse: THREE.Vector2
  } | null>(null)

  const buildScene = useCallback(() => {
    if (!canvasRef.current) return

    const W = canvasRef.current.clientWidth
    const H = canvasRef.current.clientHeight

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(W, H)
    renderer.setClearColor(0x000000, 0)
    canvasRef.current.appendChild(renderer.domElement)

    // Scene
    const scene = new THREE.Scene()
    scene.fog = new THREE.Fog(0x000000, 20, 60)

    // Camera
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 200)
    camera.position.set(0, 2, 16)
    camera.lookAt(0, -0.5, 0)

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.4)
    scene.add(ambient)
    const pointLight1 = new THREE.PointLight(0x00d4ff, 2, 30)
    pointLight1.position.set(-6, 3, 5)
    scene.add(pointLight1)
    const pointLight2 = new THREE.PointLight(0x8b5cf6, 2, 30)
    pointLight2.position.set(6, -3, 5)
    scene.add(pointLight2)
    const distLight = new THREE.PointLight(0xff6b35, 3, 15)
    distLight.position.set(8, -1.5, 3)
    scene.add(distLight)

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 5
    controls.maxDistance = 40
    controls.target.set(1, -1.5, 0)
    controls.update()

    // Helper to create nodes for one network row
    const nodeMeshes: Array<{ mesh: THREE.Mesh; layer: LayerDef }> = []

    function buildNetworkRow(
      layers: LayerDef[],
      yOffset: number,
      colorHex: number,
      emissiveHex: number
    ) {
      for (const layer of layers) {
        const yPositions = nodeYPositions(layer.count, 0.55)
        for (const yLocal of yPositions) {
          const radius = layer.isInput ? 0.38 : layer.isDistance ? 0.38 : 0.14
          const geo = new THREE.SphereGeometry(radius, 16, 16)
          const mat = new THREE.MeshStandardMaterial({
            color: layer.isInput ? 0xffffff : layer.isDistance ? 0xff6b35 : colorHex,
            emissive: layer.isInput ? 0xaaaaaa : layer.isDistance ? 0xff3300 : emissiveHex,
            emissiveIntensity: layer.isInput ? 0.6 : layer.isDistance ? 0.8 : 0.3,
            roughness: 0.2,
            metalness: 0.5,
          })
          const mesh = new THREE.Mesh(geo, mat)
          mesh.position.set(layer.x, yOffset + yLocal, 0)
          scene.add(mesh)
          nodeMeshes.push({ mesh, layer })
        }
      }
    }

    buildNetworkRow(TOP_LAYERS, 1.75, 0x00d4ff, 0x007799)
    buildNetworkRow(BOTTOM_LAYERS, -1.75, 0x8b5cf6, 0x4c1d95)
    buildNetworkRow([DISTANCE_LAYER], -0.0, 0xff6b35, 0x882200)

    // Add input glow point lights
    const inputGlow1 = new THREE.PointLight(0xffffff, 1.5, 4)
    inputGlow1.position.set(-6, 1.75, 0)
    scene.add(inputGlow1)
    const inputGlow2 = new THREE.PointLight(0xffffff, 1.5, 4)
    inputGlow2.position.set(-6, -1.75, 0)
    scene.add(inputGlow2)

    // Connections (lines between adjacent layers)
    function buildConnections(layers: LayerDef[], yOffset: number) {
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
        const mat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.06 })
        scene.add(new THREE.LineSegments(geo, mat))
      }
    }

    buildConnections(TOP_LAYERS, 1.75)
    buildConnections(BOTTOM_LAYERS, -1.75)

    // Connect last top layer to distance node
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
      const mat = new THREE.LineBasicMaterial({ color: 0xff6b35, transparent: true, opacity: 0.15 })
      scene.add(new THREE.LineSegments(geo, mat))
    }

    connectToDistance(TOP_LAYERS[TOP_LAYERS.length - 1], 1.75)
    connectToDistance(BOTTOM_LAYERS[BOTTOM_LAYERS.length - 1], -1.75)

    // Signal particles
    function buildSignalPaths(layers: LayerDef[], yOffset: number): Array<{ start: THREE.Vector3; end: THREE.Vector3 }> {
      const paths: Array<{ start: THREE.Vector3; end: THREE.Vector3 }> = []
      for (let li = 0; li < layers.length - 1; li++) {
        const fromLayer = layers[li]
        const toLayer = layers[li + 1]
        paths.push({
          start: new THREE.Vector3(fromLayer.x, yOffset, 0),
          end: new THREE.Vector3(toLayer.x, yOffset, 0),
        })
      }
      return paths
    }

    const topPaths = buildSignalPaths(TOP_LAYERS, 1.75)
    const bottomPaths = buildSignalPaths(BOTTOM_LAYERS, -1.75)
    const distancePath = {
      start: new THREE.Vector3(TOP_LAYERS[TOP_LAYERS.length - 1].x, 1.75, 0),
      end: new THREE.Vector3(DISTANCE_LAYER.x, 0, 0),
    }
    const signalPaths = [...topPaths, ...bottomPaths, distancePath]

    const signalParticles: THREE.Mesh[] = []
    const signalProgress: number[] = []

    for (let i = 0; i < signalPaths.length; i++) {
      const geo = new THREE.SphereGeometry(0.06, 8, 8)
      const isTop = i < topPaths.length
      const mat = new THREE.MeshStandardMaterial({
        color: isTop ? 0x00ffff : 0xcc99ff,
        emissive: isTop ? 0x00aaaa : 0x6633cc,
        emissiveIntensity: 1,
      })
      const mesh = new THREE.Mesh(geo, mat)
      scene.add(mesh)
      signalParticles.push(mesh)
      signalProgress.push(Math.random()) // stagger starts
    }

    // Raycaster
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()

    sceneDataRef.current = {
      renderer,
      scene,
      camera,
      controls,
      nodeMeshes,
      signalParticles,
      signalProgress,
      signalPaths,
      raycaster,
      mouse,
    }
  }, [])

  const animate = useCallback(() => {
    const data = sceneDataRef.current
    if (!data) return

    data.controls.update()

    // Move signal particles along paths
    for (let i = 0; i < data.signalParticles.length; i++) {
      data.signalProgress[i] += 0.008
      if (data.signalProgress[i] > 1) data.signalProgress[i] = 0

      const t = data.signalProgress[i]
      const path = data.signalPaths[i]
      data.signalParticles[i].position.lerpVectors(path.start, path.end, t)
    }

    // Raycasting for hover
    data.raycaster.setFromCamera(data.mouse, data.camera)
    const meshOnly = data.nodeMeshes.map((n) => n.mesh)
    const intersects = data.raycaster.intersectObjects(meshOnly)

    if (intersects.length > 0) {
      const hit = intersects[0].object as THREE.Mesh
      const nodeData = data.nodeMeshes.find((n) => n.mesh === hit)
      if (nodeData) {
        // Project 3D position to screen
        const pos = nodeData.mesh.position.clone()
        pos.project(data.camera)
        const canvas = canvasRef.current
        if (canvas) {
          const rect = canvas.getBoundingClientRect()
          const screenX = ((pos.x + 1) / 2) * rect.width
          const screenY = ((-pos.y + 1) / 2) * rect.height
          const newTooltip = {
            label: nodeData.layer.label,
            detail: nodeData.layer.detail,
            sectionId: nodeData.layer.sectionId,
            screenX,
            screenY,
          }
          if (
            !tooltipRef.current ||
            tooltipRef.current.label !== newTooltip.label
          ) {
            tooltipRef.current = newTooltip
            setTooltip(newTooltip)
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
    const W = canvas.clientWidth
    const H = canvas.clientHeight
    data.camera.aspect = W / H
    data.camera.updateProjectionMatrix()
    data.renderer.setSize(W, H)
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
        if (canvas && data.renderer.domElement.parentNode === canvas) {
          canvas.removeChild(data.renderer.domElement)
        }
      }
    }
  }, [buildScene, animate, handleResize, handleMouseMove])

  return (
    <div className="relative w-full">
      {/* Title badge */}
      <div className="flex items-center justify-center gap-3 mb-4">
        <span className="px-3 py-1 rounded-full bg-(--ollie-cyan)/10 border border-(--ollie-cyan)/30 text-(--ollie-cyan) text-xs font-bold tracking-widest uppercase">
          Interactive Network
        </span>
      </div>

      {/* Canvas container */}
      <div
        ref={canvasRef}
        className="w-full rounded-2xl overflow-hidden border border-white/10 bg-black/60"
        style={{ height: "70vh" }}
      />

      {/* Tooltip overlay */}
      {tooltip && (
        <div
          className="absolute z-20 bg-black/95 border border-(--ollie-cyan)/20 rounded-2xl px-4 py-3 max-w-[260px] shadow-2xl"
          style={{
            left: Math.min(tooltip.screenX + 16, (canvasRef.current?.clientWidth ?? 800) - 280),
            top: Math.max(tooltip.screenY - 70, 10),
            pointerEvents: 'auto',
          }}
        >
          <div className="text-(--ollie-cyan) text-[10px] font-bold tracking-widest uppercase mb-1.5">
            {tooltip.label}
          </div>
          <div className="text-white/60 text-xs leading-relaxed mb-3">{tooltip.detail}</div>
          {tooltip.sectionId && (
            <a
              href={`#${tooltip.sectionId}`}
              className="inline-flex items-center gap-1.5 text-[10px] font-bold text-(--ollie-cyan) hover:text-white transition-colors bg-(--ollie-cyan)/10 hover:bg-(--ollie-cyan)/20 border border-(--ollie-cyan)/30 rounded-lg px-2.5 py-1.5"
            >
              View full explanation â†’
            </a>
          )}
        </div>
      )}

      {/* Controls hint */}
      <p className="text-center text-white/25 text-xs mt-4 tracking-wide">
        Drag to orbit Â· Scroll to zoom Â· Hover a node for details
      </p>
    </div>
  )
}

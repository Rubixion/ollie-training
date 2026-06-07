'use client';
import { cn } from '@/lib/utils';
import { useTheme } from 'next-themes';
import { useEffect, useRef } from 'react';
import * as THREE from 'three';

// Module-level state — persists across route changes
let rendererInstance: THREE.WebGLRenderer | null = null;
let animationId: number = 0;
let count = 0;

type DottedSurfaceProps = { className?: string };

export function DottedSurface({ className, ...props }: DottedSurfaceProps) {
	const { theme } = useTheme();
	const containerRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		if (!containerRef.current) return;
		const SEPARATION = 150;
		const AMOUNTX = 40;
		const AMOUNTY = 60;
		const scene = new THREE.Scene();
		scene.fog = new THREE.Fog(0x000000, 2000, 8000);
		const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 10000);
		camera.position.set(0, 355, 1220);

		// Reuse existing renderer if possible
		if (!rendererInstance) {
			rendererInstance = new THREE.WebGLRenderer({ alpha: true, antialias: true });
		}
		const renderer = rendererInstance;
		renderer.setPixelRatio(window.devicePixelRatio);
		renderer.setSize(window.innerWidth, window.innerHeight);
		renderer.setClearColor(0x000000, 0);
		containerRef.current.appendChild(renderer.domElement);

		const positions: number[] = [];
		const colors: number[] = [];
		const geometry = new THREE.BufferGeometry();
		for (let ix = 0; ix < AMOUNTX; ix++) {
			for (let iy = 0; iy < AMOUNTY; iy++) {
				positions.push(ix * SEPARATION - (AMOUNTX * SEPARATION) / 2, 0, iy * SEPARATION - (AMOUNTY * SEPARATION) / 2);
				colors.push(0.392, 0.510, 0.824);
			}
		}
		geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
		geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
		const material = new THREE.PointsMaterial({ size: 8, vertexColors: true, transparent: true, opacity: 0.6, sizeAttenuation: true });
		const points = new THREE.Points(geometry, material);
		scene.add(points);

		let destroyed = false;
		cancelAnimationFrame(animationId);

		const animate = () => {
			if (destroyed) return;
			animationId = requestAnimationFrame(animate);
			if (document.hidden) return;
			const positionAttribute = geometry.attributes.position;
			const pos = positionAttribute.array as Float32Array;
			let i = 0;
			for (let ix = 0; ix < AMOUNTX; ix++) {
				for (let iy = 0; iy < AMOUNTY; iy++) {
					pos[i * 3 + 1] = Math.sin((ix + count) * 0.3) * 50 + Math.sin((iy + count) * 0.5) * 50;
					i++;
				}
			}
			positionAttribute.needsUpdate = true;
			renderer.render(scene, camera);
			count += 0.1;
		};

		const handleResize = () => {
			camera.aspect = window.innerWidth / window.innerHeight;
			camera.updateProjectionMatrix();
			renderer.setSize(window.innerWidth, window.innerHeight);
		};
		window.addEventListener('resize', handleResize);
		animate();

		return () => {
			destroyed = true;
			cancelAnimationFrame(animationId);
			window.removeEventListener('resize', handleResize);
			geometry.dispose();
			material.dispose();
			// Don't dispose the renderer — just detach it
			if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
				containerRef.current.removeChild(renderer.domElement);
			}
		};
	}, [theme]);

	return <div ref={containerRef} className={cn('pointer-events-none fixed inset-0 -z-10', className)} {...props} />;
}
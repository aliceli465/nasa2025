import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { createSatellite } from "./satellite.jsx";

const Earth = () => {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Create scene
    const scene = new THREE.Scene();

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 8; // Move back for larger sphere

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);

    // Add renderer to container
    mountRef.current.appendChild(renderer.domElement);

    // Add orbit controls for mouse rotation
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; // enables smooth camera movement
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 6; // Can get closer to larger sphere
    controls.maxDistance = 20; // Can zoom out further

    // Create a larger sphere with gradient material
    const geometry = new THREE.SphereGeometry(3, 32, 32);

    // Add gradient colors simulating satellite orbit (sunlight vs Earth shadow)
    const colors = [];
    const position = geometry.attributes.position;

    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i);
      const y = position.getY(i);
      const z = position.getZ(i);

      // Satellite orbit gradient: sunlight vs Earth shadow
      // Use X-axis as the orbit direction (sun side = positive X, shadow side = negative X)
      let sunlight = (x + 1) / 2; // 0 to 1 based on X position

      // Add some randomness to simulate scattered sunlight/shadow boundaries
      const noise = (Math.sin(x * 3) + Math.cos(z * 3)) * 0.1;
      sunlight = Math.max(0, Math.min(1, sunlight + noise));

      // Sun side: Bright orange/yellow (like blazing sun)
      // Shadow side: Very dark deep blue (like cold space)
      const warmColor = new THREE.Color(1.0, 0.6, 0.2); // Bright orange-yellow
      const coldColor = new THREE.Color(0.02, 0.05, 0.2); // Very dark deep blue

      // Blend between hot and cold based on sunlight
      const r = coldColor.r + (warmColor.r - coldColor.r) * sunlight;
      const g = coldColor.g + (warmColor.g - coldColor.g) * sunlight;
      const b = coldColor.b + (warmColor.b - coldColor.b) * sunlight;

      colors.push(new THREE.Color(r, g, b));
    }

    geometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(
        colors.flatMap((color) => [color.r, color.g, color.b]),
        3
      )
    );

    // Create material with just vertex colors
    const material = new THREE.MeshBasicMaterial({
      vertexColors: true,
    });

    const sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);

    // Create one satellite
    const satellite = createSatellite();
    satellite.position.set(5, 0, 0); // Position satellite to the right of Earth
    scene.add(satellite);

    // Add some lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      // Update controls
      controls.update();

      // Earth-like rotation: slow spin around Y-axis only
      sphere.rotation.y += 0.001; // Much slower, like Earth

      // Animate satellite in circular orbit around Earth
      const time = Date.now() * 0.0005; // Orbital speed
      const semiMajorAxis = 5; // Distance from Earth's center
      const semiMinorAxis = 4; // Slightly elliptical orbit

      // Circular orbit calculations
      const angle = time * 0.005; // Orbital angular velocity
      const x = semiMajorAxis * Math.cos(angle);
      const y = semiMinorAxis * Math.sin(angle);
      const z = 0; // Keep orbit in XY plane for now

      satellite.position.set(x, y, z);

      // Make satellite face the Earth
      satellite.lookAt(0, 0, 0);

      // Optional: Rotate satellite solar panels based on orbital position
      // Solar panels gradually turn as satellite orbits
      satellite.children[1].rotation.z = angle * 0.5; // Left panel
      satellite.children[2].rotation.z = angle * 0.5; // Right panel

      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      geometry.dispose();
      material.dispose();
      renderer.dispose();

      if (mountRef.current && mountRef.current.contains(renderer.domElement)) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={mountRef}
      style={{
        width: "100vw",
        height: "100vh",
        margin: 0,
        padding: 0,
      }}
    />
  );
};

export default Earth;

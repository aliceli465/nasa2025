import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";

const SatelliteVisualization = () => {
  const mountRef = useRef(null);
  const [info, setInfo] = useState({ queue: 0, completed: 0, available: 0 });
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000814);

    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      50000
    );
    camera.position.set(8000, 8000, 8000);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1.5);
    pointLight.position.set(10000, 5000, 5000);
    scene.add(pointLight);

    // Create Earth with texture-like appearance
    const earthGeometry = new THREE.SphereGeometry(4000, 64, 64);

    // Create a canvas for Earth texture
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    // Base ocean color
    ctx.fillStyle = "#1a5f7a";
    ctx.fillRect(0, 0, 512, 512);

    // Add continents (simplified green blobs)
    ctx.fillStyle = "#2d5016";

    // North America
    ctx.beginPath();
    ctx.ellipse(120, 150, 60, 80, 0.3, 0, Math.PI * 2);
    ctx.fill();

    // South America
    ctx.beginPath();
    ctx.ellipse(140, 300, 35, 70, 0.2, 0, Math.PI * 2);
    ctx.fill();

    // Europe/Africa
    ctx.beginPath();
    ctx.ellipse(280, 180, 50, 90, -0.1, 0, Math.PI * 2);
    ctx.fill();

    // Asia
    ctx.beginPath();
    ctx.ellipse(380, 140, 80, 60, 0.1, 0, Math.PI * 2);
    ctx.fill();

    // Australia
    ctx.beginPath();
    ctx.ellipse(420, 320, 35, 30, 0, 0, Math.PI * 2);
    ctx.fill();

    // Add some cloud-like white patches
    ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
    for (let i = 0; i < 20; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      const r = Math.random() * 30 + 10;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Ice caps
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.beginPath();
    ctx.ellipse(256, 30, 100, 30, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(256, 482, 100, 30, 0, 0, Math.PI * 2);
    ctx.fill();

    const texture = new THREE.CanvasTexture(canvas);

    const earthMaterial = new THREE.MeshPhongMaterial({
      map: texture,
      emissive: 0x112244,
      shininess: 25,
    });

    const earth = new THREE.Mesh(earthGeometry, earthMaterial);
    scene.add(earth);

    // Add atmosphere glow
    const atmosphereGeometry = new THREE.SphereGeometry(4100, 64, 64);
    const atmosphereMaterial = new THREE.MeshBasicMaterial({
      color: 0x4488ff,
      transparent: true,
      opacity: 0.1,
      side: THREE.BackSide,
    });
    const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    scene.add(atmosphere);

    // Agent marker on Earth surface
    const agentGeometry = new THREE.SphereGeometry(100, 16, 16);
    const agentMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const agentMarker = new THREE.Mesh(agentGeometry, agentMaterial);
    agentMarker.position.set(4000, 0, 0);
    scene.add(agentMarker);

    // Add a cone pointing outward from agent
    const coneGeometry = new THREE.ConeGeometry(80, 200, 8);
    const coneMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const agentCone = new THREE.Mesh(coneGeometry, coneMaterial);
    agentCone.position.set(4200, 0, 0);
    agentCone.rotation.z = -Math.PI / 2;
    scene.add(agentCone);

    // Satellite system - 20 satellites, most closer to Earth
    const NUM_SATELLITES = 20;
    const satellites = [];
    const satelliteMeshes = [];

    // Initialize satellites
    for (let i = 0; i < NUM_SATELLITES; i++) {
      // Most satellites closer to Earth (500-2000km), some farther
      let altitude;
      if (i < 15) {
        // 75% of satellites are close
        altitude = 500 + Math.random() * 1500;
      } else {
        // 25% are farther
        altitude = 2000 + Math.random() * 3000;
      }

      const orbitalRadius = 4000 + altitude;
      const inclination = Math.random() * Math.PI;
      const angularVelocity = 0.1 / (orbitalRadius / 4000);

      const sat = {
        orbitalRadius,
        inclination,
        currentAngle: Math.random() * 2 * Math.PI,
        angularVelocity,
        available: true,
        taskRemaining: 0,
        phase: Math.random() * 2 * Math.PI,
      };

      // Create satellite mesh - smaller for more satellites
      const satGeometry = new THREE.SphereGeometry(60, 12, 12);
      const satMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const satMesh = new THREE.Mesh(satGeometry, satMaterial);

      // Create orbital path (only show for some satellites to reduce clutter)
      if (i % 3 === 0) {
        const orbitGeometry = new THREE.BufferGeometry();
        const orbitPoints = [];
        for (let j = 0; j <= 100; j++) {
          const angle = (j / 100) * 2 * Math.PI;
          const x = orbitalRadius * Math.cos(angle);
          const y = orbitalRadius * Math.sin(angle) * Math.cos(inclination);
          const z = orbitalRadius * Math.sin(angle) * Math.sin(inclination);
          orbitPoints.push(x, y, z);
        }
        orbitGeometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(orbitPoints, 3)
        );
        const orbitMaterial = new THREE.LineBasicMaterial({
          color: 0x444444,
          opacity: 0.2,
          transparent: true,
        });
        const orbitLine = new THREE.Line(orbitGeometry, orbitMaterial);
        scene.add(orbitLine);
      }

      scene.add(satMesh);
      satellites.push(sat);
      satelliteMeshes.push(satMesh);
    }

    // Task queue simulation - 100 tasks in batches
    let taskQueue = [];
    let completedTasks = 0;
    const BATCH_SIZE = 100;

    // Generate initial tasks
    const generateTasks = (num) => {
      for (let i = 0; i < num; i++) {
        taskQueue.push({
          priority: 1 + Math.random() * 99,
          size: 0.5 + Math.random() * 4.5,
          waitTime: 0,
        });
      }
    };

    generateTasks(BATCH_SIZE);

    // Update satellite positions
    const updateSatellites = (delta) => {
      satellites.forEach((sat, idx) => {
        // Update angle
        sat.currentAngle += sat.angularVelocity * delta * 0.05;
        sat.currentAngle %= 2 * Math.PI;

        // Calculate position
        const angle = sat.currentAngle;
        const inc = sat.inclination;
        const r = sat.orbitalRadius;

        const x = r * Math.cos(angle);
        const y = r * Math.sin(angle) * Math.cos(inc);
        const z = r * Math.sin(angle) * Math.sin(inc);

        satelliteMeshes[idx].position.set(x, y, z);

        // Update task status
        if (!sat.available) {
          sat.taskRemaining -= delta * 0.01;
          if (sat.taskRemaining <= 0) {
            sat.available = true;
            sat.taskRemaining = 0;
          }
        }

        // Update color based on availability
        satelliteMeshes[idx].material.color.setHex(
          sat.available ? 0xffffff : 0xff0000
        );
      });

      // Update task wait times
      taskQueue.forEach((task) => {
        task.waitTime += delta * 0.001;
      });
    };

    // Simulate agent decisions
    const makeDecision = () => {
      if (taskQueue.length === 0 || !isRunning) return;

      // Find available satellites
      const availableSats = satellites
        .map((s, i) => ({ sat: s, idx: i }))
        .filter((s) => s.sat.available);

      if (availableSats.length > 0 && taskQueue.length > 0) {
        // Strategy: assign highest priority task to closest available satellite
        const task = taskQueue.reduce((highest, curr) =>
          curr.priority > highest.priority ? curr : highest
        );

        const agentPos = new THREE.Vector3(4000, 0, 0);
        const closestSat = availableSats.reduce(
          (closest, curr) => {
            const dist =
              satelliteMeshes[curr.idx].position.distanceTo(agentPos);
            return dist < closest.dist ? { ...curr, dist } : closest;
          },
          { dist: Infinity, idx: -1 }
        );

        if (closestSat.idx !== -1) {
          const sat = satellites[closestSat.idx];
          const duration = task.size * (1 + closestSat.dist / 10000);

          sat.available = false;
          sat.taskRemaining = duration;

          taskQueue = taskQueue.filter((t) => t !== task);
          completedTasks++;

          // Add new tasks in batches
          if (completedTasks % BATCH_SIZE === 0) {
            generateTasks(BATCH_SIZE);
          }
        }
      }
    };

    // Camera rotation
    let cameraAngle = 0;
    const cameraDistance = 12000;

    // Animation loop
    let lastDecision = 0;
    const animate = (time) => {
      requestAnimationFrame(animate);

      const delta = time - lastDecision;

      // Update satellites
      updateSatellites(1);

      // Make decision every 100ms (faster with more satellites)
      if (delta > 100 && isRunning) {
        makeDecision();
        lastDecision = time;
      }

      // Rotate Earth slowly
      earth.rotation.y += 0.0005;
      atmosphere.rotation.y += 0.0005;

      // Rotate camera
      cameraAngle += 0.001;
      camera.position.x = cameraDistance * Math.cos(cameraAngle);
      camera.position.z = cameraDistance * Math.sin(cameraAngle);
      camera.position.y = 6000;
      camera.lookAt(0, 0, 0);

      // Update info
      const availableCount = satellites.filter((s) => s.available).length;
      setInfo({
        queue: taskQueue.length,
        completed: completedTasks,
        available: availableCount,
      });

      renderer.render(scene, camera);
    };

    animate(0);

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
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, [isRunning]);

  return (
    <div
      style={{
        position: "relative",
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
      }}
    >
      <div ref={mountRef} style={{ width: "100%", height: "100%" }} />

      <div
        style={{
          position: "absolute",
          top: 20,
          left: 20,
          background: "rgba(0, 0, 0, 0.7)",
          color: "white",
          padding: "20px",
          borderRadius: "10px",
          fontFamily: "monospace",
          fontSize: "14px",
          minWidth: "250px",
        }}
      >
        <h3 style={{ margin: "0 0 15px 0", color: "#00ff00" }}>
          Satellite Task Assignment
        </h3>
        <div style={{ marginBottom: "8px" }}>
          <span style={{ color: "#aaa" }}>Tasks in Queue:</span>{" "}
          <strong>{info.queue}</strong>
        </div>
        <div style={{ marginBottom: "8px" }}>
          <span style={{ color: "#aaa" }}>Completed Tasks:</span>{" "}
          <strong>{info.completed}</strong>
        </div>
        <div style={{ marginBottom: "15px" }}>
          <span style={{ color: "#aaa" }}>Available Satellites:</span>{" "}
          <strong>{info.available}/20</strong>
        </div>

        <button
          onClick={() => setIsRunning(!isRunning)}
          style={{
            width: "100%",
            padding: "10px",
            background: isRunning ? "#ff4444" : "#44ff44",
            color: "black",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
            fontWeight: "bold",
            fontSize: "14px",
          }}
        >
          {isRunning ? "PAUSE" : "START"}
        </button>

        <div style={{ marginTop: "20px", fontSize: "12px", color: "#888" }}>
          <div style={{ marginBottom: "5px" }}>
            <span
              style={{
                display: "inline-block",
                width: "12px",
                height: "12px",
                background: "#00ff00",
                marginRight: "8px",
              }}
            ></span>
            Agent Location
          </div>
          <div style={{ marginBottom: "5px" }}>
            <span
              style={{
                display: "inline-block",
                width: "12px",
                height: "12px",
                background: "#ffffff",
                marginRight: "8px",
              }}
            ></span>
            Available Satellite
          </div>
          <div>
            <span
              style={{
                display: "inline-block",
                width: "12px",
                height: "12px",
                background: "#ff0000",
                marginRight: "8px",
              }}
            ></span>
            Busy Satellite
          </div>
        </div>
      </div>
    </div>
  );
};

export default SatelliteVisualization;

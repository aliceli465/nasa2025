import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";

const EarthSatelliteScene = ({
  width = "100%",
  height = "100vh",
  texturesPath = "./images",
  enableOrbitControls = true,
}) => {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const isDraggingRef = useRef(false);
  const previousMousePosition = useRef({ x: 0, y: 0 });
  const animationIdRef = useRef(null);
  const isVisibleRef = useRef(true);

  // State to track which satellites are blue
  const [blueSatellites, setBlueSatellites] = useState(new Set());
  // State to track downlink animations (stores animation end times)
  const [downlinkAnimations, setDownlinkAnimations] = useState(new Map());
  const downlinkAnimationsRef = useRef(new Map());
  const satellitesRef = useRef([]);
  // State to track satellite battery levels
  const [satelliteBatteries, setSatelliteBatteries] = useState({});
  const lastBatteryUpdateRef = useRef(Date.now());

  // Terminal logs state
  const [terminalLogs, setTerminalLogs] = useState([
    "System initialized...",
    "Satellites deployed successfully",
    "Battery monitoring active",
    "Orbital tracking enabled",
  ]);

  // Function to add logs to terminal
  const addLog = (message) => {
    setTerminalLogs((prev) => [...prev.slice(-9), message]); // Keep last 10 logs
  };

  // Form state
  const [formData, setFormData] = useState({
    taskName: "",
    clientName: "",
    filesNecessary: "",
    size: "",
    priority: "medium",
    description: "",
  });
  const [isFormVisible, setIsFormVisible] = useState(true);
  const downlinkBeamsRef = useRef([]);

  // Toggle satellite color
  const toggleSatelliteColor = (satelliteIndex) => {
    setBlueSatellites((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(satelliteIndex)) {
        newSet.delete(satelliteIndex);
      } else {
        newSet.add(satelliteIndex);
      }
      return newSet;
    });
  };

  // Form handlers
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    console.log("Files uploaded:", files);
    // Handle file upload logic here
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Form submitted:", formData);
    // Handle form submission logic here
  };

  // Trigger satellite downlink animation
  const triggerSatelliteDownlink = (satelliteIndex) => {
    const animationDuration = 1000; // 1 second
    const endTime = Date.now() + animationDuration;

    // Update both state and ref
    downlinkAnimationsRef.current.set(satelliteIndex, endTime);
    setDownlinkAnimations((prev) => {
      const newMap = new Map(prev);
      newMap.set(satelliteIndex, endTime);
      return newMap;
    });
  };

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      45,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      1,
      2000
    );
    scene.add(camera);
    camera.position.set(0, 35, 70);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      antialias: false, // Disable antialiasing for better performance
      powerPreference: "high-performance", // Use dedicated GPU if available
    });
    renderer.setSize(
      containerRef.current.clientWidth,
      containerRef.current.clientHeight
    );
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit pixel ratio for performance
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x8080c0, 0.8); // Much brighter ambient light for dark side
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 5.0); // Brighter sunlight
    sunLight.position.set(100, 0, 0);
    sunLight.castShadow = true;
    scene.add(sunLight);

    // Earth (reduced geometry for better performance)
    const earthGeometry = new THREE.SphereGeometry(10, 32, 32); // Reduced from 50,50 to 32,32
    const earthMaterial = new THREE.MeshPhongMaterial({
      map: new THREE.TextureLoader().load(`${texturesPath}/earthmap1k.jpg`),
      color: 0xc0c0c0, // Even brighter base color
      specular: 0x888888, // Brighter specular
      shininess: 8, // More shininess
    });
    const earth = new THREE.Mesh(earthGeometry, earthMaterial);
    earth.receiveShadow = true;
    scene.add(earth);

    // Clouds (reduced geometry)
    const cloudGeometry = new THREE.SphereGeometry(10.3, 32, 32); // Reduced from 50,50 to 32,32
    const cloudMaterial = new THREE.MeshPhongMaterial({
      map: new THREE.TextureLoader().load(`${texturesPath}/clouds_2.jpg`),
      transparent: true,
      opacity: 0.1,
    });
    const clouds = new THREE.Mesh(cloudGeometry, cloudMaterial);
    clouds.receiveShadow = true;
    scene.add(clouds);

    // Stars (reduced geometry)
    const starGeometry = new THREE.SphereGeometry(1000, 32, 32); // Reduced from 50,50 to 32,32
    const starMaterial = new THREE.MeshPhongMaterial({
      map: new THREE.TextureLoader().load(
        `${texturesPath}/galaxy_starfield.png`
      ),
      side: THREE.DoubleSide,
      shininess: 0,
    });
    const starField = new THREE.Mesh(starGeometry, starMaterial);
    scene.add(starField);

    // Animation variables
    const earthVec = new THREE.Vector3(0, 0, 0);
    let r = 30;

    // Function to create realistic satellite geometry
    const createSatelliteGeometry = () => {
      const satelliteGroup = new THREE.Group();

      // Store materials for color changing
      const materials = [];

      // Main satellite body - cylindrical with rounded edges
      const bodyGeometry = new THREE.CylinderGeometry(0.8, 0.8, 1.5, 16);
      const bodyMaterial = new THREE.MeshPhongMaterial({
        color: 0xc0c0c0, // Silver/metallic color
        emissive: 0x90ee90,
        emissiveIntensity: 0.6,
        specular: 0xffffff,
        shininess: 150,
      });
      materials.push(bodyMaterial);
      const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
      body.castShadow = true;
      body.receiveShadow = true;
      satelliteGroup.add(body);

      // Solar panel materials with rounded corners
      const solarPanelMaterial = new THREE.MeshPhongMaterial({
        color: 0x000080, // Dark blue solar panel color
        emissive: 0x90ee90,
        emissiveIntensity: 0.5,
        specular: 0x4444ff,
        shininess: 100,
        side: THREE.DoubleSide,
      });
      materials.push(solarPanelMaterial);

      // Left solar panel array (multiple segments for realism)
      const panelSegments = 3;
      for (let i = 0; i < panelSegments; i++) {
        const panelGeometry = new THREE.BoxGeometry(0.9, 0.02, 1.4);
        const panel = new THREE.Mesh(panelGeometry, solarPanelMaterial);
        panel.position.set(-2.5 - i * 0.95, 0, 0);
        panel.castShadow = true;
        panel.receiveShadow = true;
        satelliteGroup.add(panel);
      }

      // Right solar panel array
      for (let i = 0; i < panelSegments; i++) {
        const panelGeometry = new THREE.BoxGeometry(0.9, 0.02, 1.4);
        const panel = new THREE.Mesh(panelGeometry, solarPanelMaterial);
        panel.position.set(2.5 + i * 0.95, 0, 0);
        panel.castShadow = true;
        panel.receiveShadow = true;
        satelliteGroup.add(panel);
      }

      // Solar panel support arms - smoother cylinders
      const armMaterial = new THREE.MeshPhongMaterial({
        color: 0x808080, // Gray metal color
        emissive: 0x90ee90,
        emissiveIntensity: 0.5,
        specular: 0xffffff,
        shininess: 100,
      });
      materials.push(armMaterial);

      // Left arm
      const leftArmGeometry = new THREE.CylinderGeometry(0.08, 0.08, 1.8, 8);
      const leftArm = new THREE.Mesh(leftArmGeometry, armMaterial);
      leftArm.position.set(-1.4, 0, 0);
      leftArm.rotation.z = Math.PI / 2;
      satelliteGroup.add(leftArm);

      // Right arm
      const rightArmGeometry = new THREE.CylinderGeometry(0.08, 0.08, 1.8, 8);
      const rightArm = new THREE.Mesh(rightArmGeometry, armMaterial);
      rightArm.position.set(1.4, 0, 0);
      rightArm.rotation.z = Math.PI / 2;
      satelliteGroup.add(rightArm);

      // Communication dish - smoother sphere segment
      const dishGeometry = new THREE.SphereGeometry(
        0.4,
        16,
        8,
        0,
        Math.PI * 2,
        0,
        Math.PI / 2
      );
      const dishMaterial = new THREE.MeshPhongMaterial({
        color: 0xe0e0e0, // Light gray/white dish color
        emissive: 0x90ee90,
        emissiveIntensity: 0.7,
        specular: 0xffffff,
        shininess: 200,
        side: THREE.DoubleSide,
      });
      materials.push(dishMaterial);
      const dish = new THREE.Mesh(dishGeometry, dishMaterial);
      dish.position.set(0, 1, 0);
      dish.rotation.x = Math.PI;
      satelliteGroup.add(dish);

      // Antenna mast
      const antennaGeometry = new THREE.CylinderGeometry(0.03, 0.03, 0.6, 8);
      const antennaMaterial = new THREE.MeshPhongMaterial({
        color: 0xa0a0a0, // Light gray metal color
        emissive: 0x90ee90,
        emissiveIntensity: 0.7,
        specular: 0xffffff,
        shininess: 150,
      });
      materials.push(antennaMaterial);
      const antenna = new THREE.Mesh(antennaGeometry, antennaMaterial);
      antenna.position.set(0, 0.5, 0);
      satelliteGroup.add(antenna);

      // Small sensor spheres for detail
      const sensorMaterial = new THREE.MeshPhongMaterial({
        color: 0x606060, // Dark gray sensor color
        emissive: 0x90ee90,
        emissiveIntensity: 0.8,
        specular: 0xffffff,
        shininess: 200,
      });
      materials.push(sensorMaterial);

      const sensorGeometry = new THREE.SphereGeometry(0.1, 8, 6);
      const sensor1 = new THREE.Mesh(sensorGeometry, sensorMaterial);
      sensor1.position.set(0.5, 0.3, 0.5);
      satelliteGroup.add(sensor1);

      const sensor2 = new THREE.Mesh(sensorGeometry, sensorMaterial);
      sensor2.position.set(-0.5, 0.3, 0.5);
      satelliteGroup.add(sensor2);

      // Store materials reference for color changing
      satelliteGroup.userData.materials = materials;

      return satelliteGroup;
    };

    // Satellites (6 equally spaced)

    // First satellite (0 degrees)
    const satellite1 = createSatelliteGeometry();
    satellite1.position.set(r, 0, 0);
    scene.add(satellite1);

    // Second satellite (60 degrees)
    const satellite2 = createSatelliteGeometry();
    satellite2.position.set(
      r * Math.cos(Math.PI / 3),
      0,
      r * Math.sin(Math.PI / 3)
    );
    scene.add(satellite2);

    // Third satellite (120 degrees)
    const satellite3 = createSatelliteGeometry();
    satellite3.position.set(
      r * Math.cos((2 * Math.PI) / 3),
      0,
      r * Math.sin((2 * Math.PI) / 3)
    );
    scene.add(satellite3);

    // Fourth satellite (180 degrees)
    const satellite4 = createSatelliteGeometry();
    satellite4.position.set(-r, 0, 0);
    scene.add(satellite4);

    // Fifth satellite (240 degrees)
    const satellite5 = createSatelliteGeometry();
    satellite5.position.set(
      r * Math.cos((4 * Math.PI) / 3),
      0,
      r * Math.sin((4 * Math.PI) / 3)
    );
    scene.add(satellite5);

    // Sixth satellite (300 degrees)
    const satellite6 = createSatelliteGeometry();
    satellite6.position.set(
      r * Math.cos((5 * Math.PI) / 3),
      0,
      r * Math.sin((5 * Math.PI) / 3)
    );
    scene.add(satellite6);

    // Store satellite references
    satellitesRef.current = [
      satellite1,
      satellite2,
      satellite3,
      satellite4,
      satellite5,
      satellite6,
    ];

    // Add simple text label to first satellite
    const createTextSprite = (text) => {
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      canvas.width = 64;
      canvas.height = 64;

      // Fill with white background
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);

      context.font = "bold 32px Arial";
      context.fillStyle = "#000000";
      context.textAlign = "center";
      context.textBaseline = "middle";

      context.fillText(text, canvas.width / 2, canvas.height / 2);

      const texture = new THREE.CanvasTexture(canvas);
      const spriteMaterial = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
      });
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.scale.set(3, 3, 1);

      return sprite;
    };

    // Add labels to all satellites
    const satelliteLabels = [];
    satellitesRef.current.forEach((satellite, index) => {
      const label = createTextSprite((index + 1).toString());
      label.position.y = 3; // Position above the satellite
      satellite.add(label);
      satelliteLabels.push(label);
    });

    // Initialize battery levels to 100%
    const initialBatteries = {};
    satellitesRef.current.forEach((_, index) => {
      initialBatteries[index] = 100;
    });
    setSatelliteBatteries(initialBatteries);

    // Create downlink beams
    const createDownlinkBeam = () => {
      const beamGroup = new THREE.Group();

      // Main beam line - using BoxGeometry for better control
      const beamGeometry = new THREE.BoxGeometry(0.1, 0.1, 1);
      const beamMaterial = new THREE.MeshPhongMaterial({
        color: 0x00ffff,
        emissive: 0x00ffff,
        emissiveIntensity: 0.8,
        transparent: true,
        opacity: 0.7,
      });

      const beam = new THREE.Mesh(beamGeometry, beamMaterial);
      beamGroup.add(beam);

      return beamGroup;
    };

    // Initialize downlink beams array
    downlinkBeamsRef.current = satellitesRef.current.map(() => {
      const beam = createDownlinkBeam();
      beam.visible = false;
      scene.add(beam);
      return beam;
    });

    let theta = 0;
    const dTheta = (2 * Math.PI) / 3000; // Even slower orbital speed
    let dx = enableOrbitControls ? 0 : 0.01;
    let dy = enableOrbitControls ? 0 : -0.01;
    let dz = enableOrbitControls ? 0 : -0.05;

    // Mouse control variables
    const spherical = new THREE.Spherical();
    const rotationSpeed = 0.005;

    // Mouse event handlers
    const handleMouseDown = (event) => {
      isDraggingRef.current = true;
      previousMousePosition.current = {
        x: event.clientX,
        y: event.clientY,
      };
    };

    const handleMouseMove = (event) => {
      if (!isDraggingRef.current || !enableOrbitControls) return;

      const deltaX = event.clientX - previousMousePosition.current.x;
      const deltaY = event.clientY - previousMousePosition.current.y;

      // Convert camera position to spherical coordinates
      const offset = new THREE.Vector3().copy(camera.position).sub(earthVec);
      spherical.setFromVector3(offset);

      // Rotate horizontally and vertically
      spherical.theta -= deltaX * rotationSpeed;
      spherical.phi -= deltaY * rotationSpeed;

      // Limit vertical rotation
      spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

      // Convert back to Cartesian coordinates
      offset.setFromSpherical(spherical);
      camera.position.copy(earthVec).add(offset);

      previousMousePosition.current = {
        x: event.clientX,
        y: event.clientY,
      };
    };

    const handleMouseUp = () => {
      isDraggingRef.current = false;
    };

    const handleMouseWheel = (event) => {
      if (!enableOrbitControls) return;
      event.preventDefault();

      const offset = new THREE.Vector3().copy(camera.position).sub(earthVec);
      const distance = offset.length();
      const newDistance = distance * (1 + event.deltaY * 0.001);

      // Limit zoom distance
      const clampedDistance = Math.max(15, Math.min(200, newDistance));
      offset.setLength(clampedDistance);
      camera.position.copy(earthVec).add(offset);
    };

    // Add event listeners
    if (enableOrbitControls) {
      renderer.domElement.addEventListener("mousedown", handleMouseDown);
      renderer.domElement.addEventListener("mousemove", handleMouseMove);
      renderer.domElement.addEventListener("mouseup", handleMouseUp);
      renderer.domElement.addEventListener("mouseleave", handleMouseUp);
      renderer.domElement.addEventListener("wheel", handleMouseWheel, {
        passive: false,
      });
    }

    // Visibility observer to pause animation when not visible
    const observer = new IntersectionObserver(
      (entries) => {
        isVisibleRef.current = entries[0].isIntersecting;
      },
      { threshold: 0.1 }
    );
    observer.observe(containerRef.current);

    // Pause animation when page is not in focus
    const handleVisibilityChange = () => {
      if (document.hidden) {
        isVisibleRef.current = false;
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);

      // Only animate if component is visible
      if (!isVisibleRef.current) return;

      earth.rotation.y += 0.0009;
      clouds.rotation.y += 0.00005;

      // Satellite orbits
      theta += dTheta;

      // First satellite orbit (0 degrees)
      satellite1.position.x = r * Math.cos(theta);
      satellite1.position.z = r * Math.sin(theta);

      // Second satellite orbit (60 degrees)
      satellite2.position.x = r * Math.cos(theta + Math.PI / 3);
      satellite2.position.z = r * Math.sin(theta + Math.PI / 3);

      // Third satellite orbit (120 degrees)
      satellite3.position.x = r * Math.cos(theta + (2 * Math.PI) / 3);
      satellite3.position.z = r * Math.sin(theta + (2 * Math.PI) / 3);

      // Fourth satellite orbit (180 degrees)
      satellite4.position.x = r * Math.cos(theta + Math.PI);
      satellite4.position.z = r * Math.sin(theta + Math.PI);

      // Fifth satellite orbit (240 degrees)
      satellite5.position.x = r * Math.cos(theta + (4 * Math.PI) / 3);
      satellite5.position.z = r * Math.sin(theta + (4 * Math.PI) / 3);

      // Sixth satellite orbit (300 degrees)
      satellite6.position.x = r * Math.cos(theta + (5 * Math.PI) / 3);
      satellite6.position.z = r * Math.sin(theta + (5 * Math.PI) / 3);

      // Update satellite orientations to face Earth
      const satellites = [
        satellite1,
        satellite2,
        satellite3,
        satellite4,
        satellite5,
        satellite6,
      ];

      // Update battery levels based on position
      const currentTime = Date.now();
      const timeSinceLastUpdate = currentTime - lastBatteryUpdateRef.current;

      if (timeSinceLastUpdate > 300) {
        // Update every 300ms
        setSatelliteBatteries((prevBatteries) => {
          const newBatteries = { ...prevBatteries };

          satellites.forEach((satellite, index) => {
            const currentBattery = newBatteries[index] || 100;

            if (satellite.position.x < -5) {
              // In shadow - discharge by 1%
              newBatteries[index] = Math.max(0, currentBattery - 1);
            } else if (satellite.position.x > 5 && currentBattery < 100) {
              // In sunlight and not full - charge by 1%
              newBatteries[index] = Math.min(100, currentBattery + 1);
            }
          });

          lastBatteryUpdateRef.current = currentTime;
          return newBatteries;
        });
      }

      satellites.forEach((satellite, index) => {
        // Make satellites face Earth
        satellite.lookAt(earthVec);

        // Change satellite glow based on Earth's shadow with smooth transition
        const materials = satellite.userData.materials;
        if (materials) {
          // Check if this satellite should be blue
          const isBlue = satellite.userData.isBlue || false;

          if (isBlue) {
            // Blue color for toggled satellites
            const blueColor = new THREE.Color(0x0080ff);
            materials.forEach((material) => {
              material.emissive = blueColor;
              material.emissiveIntensity = 1.2; // Bright blue glow
            });
          } else {
            // Satellites stay green all the time
            const greenColor = new THREE.Color(0x90ee90);
            materials.forEach((material) => {
              material.emissive = greenColor;
              material.emissiveIntensity = 1.0;
            });
          }
        }
      });

      // Update downlink beams
      const beamUpdateTime = Date.now();
      downlinkBeamsRef.current.forEach((beam, index) => {
        const satellite = satellitesRef.current[index];

        // Check if this satellite has an active animation
        let isAnimating = false;
        let animationEndTime = null;

        // Access the latest animations through ref
        const endTime = downlinkAnimationsRef.current.get(index);
        if (endTime && beamUpdateTime < endTime) {
          isAnimating = true;
          animationEndTime = endTime;
        }

        if (isAnimating && satellite) {
          beam.visible = true;

          // Calculate direction and distance from satellite to Earth center
          const satellitePos = satellite.position.clone();
          const direction = earthVec.clone().sub(satellitePos);
          const distance = direction.length();

          // Position beam group at satellite location
          beam.position.copy(satellitePos);

          // Point beam toward Earth
          beam.lookAt(earthVec);

          // Calculate animation progress (0 to 1)
          const animationDuration = 1000; // 1 second
          const timeLeft = animationEndTime - currentTime;
          const progress = 1 - timeLeft / animationDuration;

          // Animate beam appearance with fade in/out
          let opacity = 0.7;
          if (progress < 0.2) {
            // Fade in
            opacity = (progress / 0.2) * 0.7;
          } else if (progress > 0.8) {
            // Fade out
            opacity = ((1 - progress) / 0.2) * 0.7;
          }

          // Scale and position the main beam box to extend toward Earth
          const mainBeam = beam.children[0];
          mainBeam.scale.set(1, 1, distance * 0.8); // Scale Z to reach toward Earth
          mainBeam.position.set(0, 0, distance * 0.4); // Move beam toward Earth (positive Z)
          mainBeam.material.opacity = opacity;
        } else {
          beam.visible = false;
        }
      });

      // Camera flyby (only if orbit controls disabled)
      if (!enableOrbitControls) {
        if (camera.position.z < 0) {
          dx *= -1;
        }
        camera.position.x += dx;
        camera.position.y += dy;
        camera.position.z += dz;
      }

      camera.lookAt(earthVec);
      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      // Cancel animation loop
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }

      // Disconnect observer
      observer.disconnect();

      // Remove event listeners
      window.removeEventListener("resize", handleResize);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (enableOrbitControls && renderer.domElement) {
        renderer.domElement.removeEventListener("mousedown", handleMouseDown);
        renderer.domElement.removeEventListener("mousemove", handleMouseMove);
        renderer.domElement.removeEventListener("mouseup", handleMouseUp);
        renderer.domElement.removeEventListener("mouseleave", handleMouseUp);
        renderer.domElement.removeEventListener("wheel", handleMouseWheel);
      }

      // Remove renderer from DOM
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }

      // Dispose of all materials and geometries
      renderer.dispose();
      earthGeometry.dispose();
      earthMaterial.dispose();
      cloudGeometry.dispose();
      cloudMaterial.dispose();
      starGeometry.dispose();
      starMaterial.dispose();

      // Dispose of satellite geometries and materials
      const satellites = [
        satellite1,
        satellite2,
        satellite3,
        satellite4,
        satellite5,
        satellite6,
      ];
      satellites.forEach((satellite) => {
        satellite.traverse((child) => {
          if (child.geometry) child.geometry.dispose();
          if (child.material) child.material.dispose();
        });
      });
    };
  }, [texturesPath, enableOrbitControls]);

  // Separate effect to handle color changes without rebuilding the scene
  useEffect(() => {
    if (satellitesRef.current.length === 0) return;

    // Update satellite colors based on blueSatellites state
    satellitesRef.current.forEach((satellite, index) => {
      if (satellite && satellite.userData.materials) {
        satellite.userData.isBlue = blueSatellites.has(index);
      }
    });
  }, [blueSatellites]);

  // Clean up expired animations
  useEffect(() => {
    const cleanup = () => {
      const currentTime = Date.now();

      // Clean up ref
      const newRefMap = new Map();
      for (const [key, endTime] of downlinkAnimationsRef.current) {
        if (currentTime < endTime) {
          newRefMap.set(key, endTime);
        }
      }
      downlinkAnimationsRef.current = newRefMap;

      // Clean up state
      setDownlinkAnimations((prev) => {
        const newMap = new Map();
        for (const [key, endTime] of prev) {
          if (currentTime < endTime) {
            newMap.set(key, endTime);
          }
        }
        return newMap;
      });
    };

    const interval = setInterval(cleanup, 1000); // Clean up every second
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ position: "relative", width, height }}>
      {/* Three.js Scene - Full width */}
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "100%",
          backgroundColor: "black",
          overflow: "hidden",
          cursor: enableOrbitControls ? "grab" : "default",
        }}
      />

      {/* Color Legend */}
      <div className="mt-16 absolute top-4 left-4 z-20 bg-black/80 backdrop-blur-md rounded-lg p-4 border border-white/20">
        <h4 className="text-[#dfdff2] text-sm font-bold mb-3">
          Satellite Status
        </h4>
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]"></div>
            <span className="text-[#dfdff2] text-xs">Available</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]"></div>
            <span className="text-[#dfdff2] text-xs">In Use</span>
          </div>
        </div>
      </div>

      {/* Right Column Container */}
      <div className="mt-8 absolute top-16 right-4 w-80 h-3/4 z-50 flex flex-col">
        {/* Terminal Component - Top */}
        <div className="bg-black/90 backdrop-blur-md rounded-lg p-3 border border-white/20 mb-4 flex-shrink-0">
          <div className="flex items-center gap-2 mb-2">
            <div className="flex gap-1">
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
              <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
              <div className="w-2 h-2 rounded-full bg-green-500"></div>
            </div>
            <span className="text-[#dfdff2] text-xs font-mono">
              System Logs
            </span>
          </div>
          <div className="h-32 overflow-y-auto bg-black/50 rounded p-2 font-mono text-xs">
            {terminalLogs.map((log, index) => (
              <div key={index} className="text-green-400 mb-1">
                <span className="text-gray-500">
                  [{new Date().toLocaleTimeString()}]
                </span>{" "}
                {log}
              </div>
            ))}
          </div>
        </div>

        {/* Form Panel - Bottom */}
        <div className="flex-1 bg-black/90 backdrop-blur-md border-l border-white/10 p-7 overflow-y-auto min-h-0">
          {/* Client Form */}
          <div>
            <div className="mt-8 flex justify-between items-center mb-5">
              <h3 className="text-[#dfdff2] text-base font-bold m-0">
                Submit Computing Task
              </h3>
              <button
                onClick={() => setIsFormVisible(!isFormVisible)}
                className="bg-none border-none text-[#dfdff2] cursor-pointer text-lg p-1"
              >
                {isFormVisible ? "−" : "+"}
              </button>
            </div>

            {isFormVisible && (
              <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                <div>
                  <label className="text-[#dfdff2] text-sm mb-1 block">
                    Task Name
                  </label>
                  <input
                    type="text"
                    name="taskName"
                    value={formData.taskName}
                    onChange={handleInputChange}
                    placeholder="e.g., AI Model Training"
                    className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                    required
                  />
                </div>

                <div>
                  <label className="text-[#dfdff2] text-sm mb-1 block">
                    Client Name
                  </label>
                  <input
                    type="text"
                    name="clientName"
                    value={formData.clientName}
                    onChange={handleInputChange}
                    placeholder="Your company name"
                    className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                    required
                  />
                </div>

                <div>
                  <label className="text-[#dfdff2] text-sm mb-1 block">
                    Files Necessary
                  </label>
                  <input
                    type="text"
                    name="filesNecessary"
                    value={formData.filesNecessary}
                    onChange={handleInputChange}
                    placeholder="e.g., dataset.csv, model.py"
                    className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                  />
                </div>

                <div>
                  <label className="text-[#dfdff2] text-sm mb-1 block">
                    Data Size
                  </label>
                  <select
                    name="size"
                    value={formData.size}
                    onChange={handleInputChange}
                    className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                    required
                  >
                    <option value="">Select size</option>
                    <option value="small">Small (&lt; 1GB)</option>
                    <option value="medium">Medium (1-10GB)</option>
                    <option value="large">Large (10-100GB)</option>
                    <option value="xlarge">Extra Large (&gt; 100GB)</option>
                  </select>
                </div>

                <div>
                  <label className="text-[#dfdff2] text-sm mb-1 block">
                    Priority
                  </label>
                  <select
                    name="priority"
                    value={formData.priority}
                    onChange={handleInputChange}
                    className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="urgent">Urgent</option>
                  </select>
                </div>

                <div>
                  <label className="text-[#dfdff2] text-sm mb-1 block">
                    Description
                  </label>
                  <textarea
                    name="description"
                    value={formData.description}
                    onChange={handleInputChange}
                    placeholder="Describe your computing task..."
                    rows="3"
                    className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm resize-y"
                  />
                </div>

                <div>
                  <label className="text-[#dfdff2] text-sm mb-1 block">
                    Upload Files
                  </label>
                  <input
                    type="file"
                    multiple
                    onChange={handleFileUpload}
                    className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                  />
                </div>

                <button
                  type="submit"
                  className="w-full p-3 bg-[#dfdff2] text-black border-none rounded text-base font-bold cursor-pointer transition-all duration-300 hover:bg-[#c0c0d0] hover:-translate-y-0.5"
                >
                  Submit Task
                </button>
              </form>
            )}
          </div>
        </div>
      </div>

      {/* Satellite Battery Table */}
      <div className="absolute bottom-4 left-4 z-20 bg-black/80 backdrop-blur-md rounded-lg p-4 border border-white/20">
        <h4 className="text-[#dfdff2] text-sm font-bold mb-3">
          Satellite Battery Levels
        </h4>
        <div className="space-y-2">
          {satellitesRef.current.map((satellite, index) => {
            // Get actual battery level from state
            const batteryLevel = satelliteBatteries[index] || 100;
            let batteryColor = "bg-gray-500";

            // Determine color based on battery level only
            if (batteryLevel >= 80) {
              // High battery - green
              batteryColor = "bg-green-500";
            } else if (batteryLevel >= 50) {
              // Medium battery - yellow
              batteryColor = "bg-yellow-500";
            } else {
              // Low battery - red
              batteryColor = "bg-red-500";
            }

            return (
              <div
                key={index}
                className="flex items-center justify-between gap-3 min-w-[200px]"
              >
                <span className="text-[#dfdff2] text-xs font-medium">
                  Satellite {index + 1}
                </span>
                <div className="flex items-center gap-2">
                  <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${batteryColor} transition-all duration-300`}
                      style={{ width: `${batteryLevel}%` }}
                    ></div>
                  </div>
                  <span className="text-[#dfdff2] text-xs w-8 text-right">
                    {Math.round(batteryLevel)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Form Panel - 1/4 width */}
      <div className="w-1/4 h-full bg-black/90 backdrop-blur-md border-l border-white/10 p-5 overflow-y-auto">
        {/* <div className="mt-16 mb-8">
          <h3 className="text-[#dfdff2] text-base font-bold mb-4">
            Satellite Controls
          </h3>
          <div className="flex flex-col gap-2">
            {[0, 1, 2, 3, 4, 5].map((index) => (
              <div key={index} className="flex gap-2">
                <button
                  onClick={() => toggleSatelliteColor(index)}
                  className={`px-3 py-2 text-white border-none rounded cursor-pointer text-xs font-bold transition-all duration-300 hover:scale-105 flex-1 ${
                    blueSatellites.has(index)
                      ? "bg-blue-500 shadow-[0_0_10px_rgba(0,128,255,0.5)] hover:shadow-[0_0_15px_rgba(0,128,255,0.7)]"
                      : "bg-gray-700 shadow-[0_2px_4px_rgba(0,0,0,0.2)] hover:shadow-[0_4px_8px_rgba(0,0,0,0.3)]"
                  }`}
                >
                  Satellite {index + 1}
                </button>
                <button
                  onClick={() => triggerSatelliteDownlink(index)}
                  className="px-3 py-2 bg-cyan-500 text-white border-none rounded cursor-pointer text-xs font-bold transition-all duration-300 hover:scale-105 hover:bg-cyan-400 shadow-[0_2px_4px_rgba(0,255,255,0.2)] hover:shadow-[0_4px_8px_rgba(0,255,255,0.4)]"
                >
                  Downlink
                </button>
              </div>
            ))}
          </div>
        </div> */}

        {/* Client Form */}
        <div>
          <div className="mt-16 flex justify-between items-center mb-5">
            <h3 className="text-[#dfdff2] text-base font-bold m-0">
              Submit Computing Task
            </h3>
            {/* <button
              onClick={() => setIsFormVisible(!isFormVisible)}
              className="bg-none border-none text-[#dfdff2] cursor-pointer text-lg p-1"
            >
              {isFormVisible ? "−" : "+"}
            </button> */}
          </div>

          {isFormVisible && (
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div>
                <label className="text-[#dfdff2] text-sm mb-1 block">
                  Task Name
                </label>
                <input
                  type="text"
                  name="taskName"
                  value={formData.taskName}
                  onChange={handleInputChange}
                  placeholder="e.g., AI Model Training"
                  className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                  required
                />
              </div>

              <div>
                <label className="text-[#dfdff2] text-sm mb-1 block">
                  Client Name
                </label>
                <input
                  type="text"
                  name="clientName"
                  value={formData.clientName}
                  onChange={handleInputChange}
                  placeholder="Your company name"
                  className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                  required
                />
              </div>

              <div>
                <label className="text-[#dfdff2] text-sm mb-1 block">
                  Files Necessary
                </label>
                <input
                  type="text"
                  name="filesNecessary"
                  value={formData.filesNecessary}
                  onChange={handleInputChange}
                  placeholder="e.g., dataset.csv, model.py"
                  className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                />
              </div>

              <div>
                <label className="text-[#dfdff2] text-sm mb-1 block">
                  Data Size
                </label>
                <select
                  name="size"
                  value={formData.size}
                  onChange={handleInputChange}
                  className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                  required
                >
                  <option value="">Select size</option>
                  <option value="small">Small (&lt; 1GB)</option>
                  <option value="medium">Medium (1-10GB)</option>
                  <option value="large">Large (10-100GB)</option>
                  <option value="xlarge">Extra Large (&gt; 100GB)</option>
                </select>
              </div>

              <div>
                <label className="text-[#dfdff2] text-sm mb-1 block">
                  Priority
                </label>
                <select
                  name="priority"
                  value={formData.priority}
                  onChange={handleInputChange}
                  className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="urgent">Urgent</option>
                </select>
              </div>

              <div>
                <label className="text-[#dfdff2] text-sm mb-1 block">
                  Description
                </label>
                <textarea
                  name="description"
                  value={formData.description}
                  onChange={handleInputChange}
                  placeholder="Describe your computing task..."
                  rows="3"
                  className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm resize-y"
                />
              </div>

              <div>
                <label className="text-[#dfdff2] text-sm mb-1 block">
                  Upload Files
                </label>
                <input
                  type="file"
                  multiple
                  onChange={handleFileUpload}
                  className="w-full p-2.5 rounded border border-white/20 bg-white/10 text-[#dfdff2] text-sm"
                />
              </div>

              <button
                type="submit"
                className="w-full p-3 bg-[#dfdff2] text-black border-none rounded text-base font-bold cursor-pointer transition-all duration-300 hover:bg-[#c0c0d0] hover:-translate-y-0.5"
              >
                Submit Task
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};

export default EarthSatelliteScene;

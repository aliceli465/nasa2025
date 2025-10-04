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
  const satellitesRef = useRef([]);

  // Form state
  const [formData, setFormData] = useState({
    taskName: "",
    clientName: "",
    filesNecessary: "",
    size: "",
    priority: "medium",
    description: "",
  });
  const [isFormVisible, setIsFormVisible] = useState(false);

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

    let theta = 0;
    const dTheta = (2 * Math.PI) / 2000; // Slower orbital speed
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
            // Normal red/green transition
            // Calculate transition factor based on position
            // Use a smooth transition zone around x = 0
            const transitionZone = 5; // Width of transition zone
            let transitionFactor;

            if (satellite.position.x < -transitionZone) {
              transitionFactor = 0; // Fully in shadow (red)
            } else if (satellite.position.x > transitionZone) {
              transitionFactor = 1; // Fully in sunlight (green)
            } else {
              // Smooth transition using cosine interpolation
              transitionFactor =
                (Math.sin(
                  ((satellite.position.x / transitionZone) * Math.PI) / 2
                ) +
                  1) /
                2;
            }

            // Interpolate between red and green
            const redColor = new THREE.Color(0xfa1937);
            const greenColor = new THREE.Color(0x90ee90);
            const emissiveColor = new THREE.Color();
            emissiveColor.lerpColors(redColor, greenColor, transitionFactor);

            // Interpolate intensity
            const shadowIntensity = 0.8;
            const sunlightIntensity = 1.0;
            const emissiveIntensity =
              shadowIntensity +
              (sunlightIntensity - shadowIntensity) * transitionFactor;

            materials.forEach((material) => {
              material.emissive = emissiveColor;
              material.emissiveIntensity = emissiveIntensity;
            });
          }
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

  return (
    <div style={{ position: "relative", width, height, display: "flex" }}>
      {/* Three.js Scene - 3/4 width */}
      <div
        ref={containerRef}
        style={{
          width: "75%",
          height: "100%",
          backgroundColor: "black",
          overflow: "hidden",
          cursor: enableOrbitControls ? "grab" : "default",
        }}
      />

      {/* Form Panel - 1/4 width */}
      <div className="w-1/4 h-full bg-black/90 backdrop-blur-md border-l border-white/10 p-5 overflow-y-auto">
        <div className=" mt-16 mb-8">
          <div className="flex flex-col gap-2">
            {[0, 1, 2, 3, 4, 5].map((index) => (
              <button
                key={index}
                onClick={() => toggleSatelliteColor(index)}
                className={`px-3 py-2 text-white border-none rounded cursor-pointer text-xs font-bold transition-all duration-300 hover:scale-105 ${
                  blueSatellites.has(index)
                    ? "bg-blue-500 shadow-[0_0_10px_rgba(0,128,255,0.5)] hover:shadow-[0_0_15px_rgba(0,128,255,0.7)]"
                    : "bg-gray-700 shadow-[0_2px_4px_rgba(0,0,0,0.2)] hover:shadow-[0_4px_8px_rgba(0,0,0,0.3)]"
                }`}
              >
                Satellite {index + 1}
              </button>
            ))}
          </div>
        </div>

        {/* Client Form */}
        <div>
          <div className="flex justify-between items-center mb-5">
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
  );
};

export default EarthSatelliteScene;

import React, { useEffect, useRef } from "react";
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

    // Satellites (6 equally spaced)
    const satelliteGeometry = new THREE.BoxGeometry(3, 1, 1);

    // First satellite (0 degrees)
    const satellite1Material = new THREE.MeshBasicMaterial({
      color: 0x90ee90,
    });
    const satellite1 = new THREE.Mesh(satelliteGeometry, satellite1Material);
    satellite1.position.set(r, 0, 0);
    scene.add(satellite1);

    // Second satellite (60 degrees)
    const satellite2Material = new THREE.MeshBasicMaterial({
      color: 0x90ee90,
    });
    const satellite2 = new THREE.Mesh(satelliteGeometry, satellite2Material);
    satellite2.position.set(
      r * Math.cos(Math.PI / 3),
      0,
      r * Math.sin(Math.PI / 3)
    );
    scene.add(satellite2);

    // Third satellite (120 degrees)
    const satellite3Material = new THREE.MeshBasicMaterial({
      color: 0x90ee90,
    });
    const satellite3 = new THREE.Mesh(satelliteGeometry, satellite3Material);
    satellite3.position.set(
      r * Math.cos((2 * Math.PI) / 3),
      0,
      r * Math.sin((2 * Math.PI) / 3)
    );
    scene.add(satellite3);

    // Fourth satellite (180 degrees)
    const satellite4Material = new THREE.MeshBasicMaterial({
      color: 0x90ee90,
    });
    const satellite4 = new THREE.Mesh(satelliteGeometry, satellite4Material);
    satellite4.position.set(-r, 0, 0);
    scene.add(satellite4);

    // Fifth satellite (240 degrees)
    const satellite5Material = new THREE.MeshBasicMaterial({
      color: 0x90ee90,
    });
    const satellite5 = new THREE.Mesh(satelliteGeometry, satellite5Material);
    satellite5.position.set(
      r * Math.cos((4 * Math.PI) / 3),
      0,
      r * Math.sin((4 * Math.PI) / 3)
    );
    scene.add(satellite5);

    // Sixth satellite (300 degrees)
    const satellite6Material = new THREE.MeshBasicMaterial({
      color: 0x90ee90,
    });
    const satellite6 = new THREE.Mesh(satelliteGeometry, satellite6Material);
    satellite6.position.set(
      r * Math.cos((5 * Math.PI) / 3),
      0,
      r * Math.sin((5 * Math.PI) / 3)
    );
    scene.add(satellite6);
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

      // Change satellite colors based on Earth's shadow
      if (satellite1.position.x < 0) {
        satellite1Material.color.setHex(0xfa1937); // Red in shadow
      } else {
        satellite1Material.color.setHex(0x90ee90); // Green in sunlight
      }

      if (satellite2.position.x < 0) {
        satellite2Material.color.setHex(0xfa1937); // Red in shadow
      } else {
        satellite2Material.color.setHex(0x90ee90); // Green in sunlight
      }

      if (satellite3.position.x < 0) {
        satellite3Material.color.setHex(0xfa1937); // Red in shadow
      } else {
        satellite3Material.color.setHex(0x90ee90); // Green in sunlight
      }

      if (satellite4.position.x < 0) {
        satellite4Material.color.setHex(0xfa1937); // Red in shadow
      } else {
        satellite4Material.color.setHex(0x90ee90); // Green in sunlight
      }

      if (satellite5.position.x < 0) {
        satellite5Material.color.setHex(0xfa1937); // Red in shadow
      } else {
        satellite5Material.color.setHex(0x90ee90); // Green in sunlight
      }

      if (satellite6.position.x < 0) {
        satellite6Material.color.setHex(0xfa1937); // Red in shadow
      } else {
        satellite6Material.color.setHex(0x90ee90); // Green in sunlight
      }

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
      satelliteGeometry.dispose();
      satellite1Material.dispose();
      satellite2Material.dispose();
      satellite3Material.dispose();
      satellite4Material.dispose();
      satellite5Material.dispose();
      satellite6Material.dispose();
    };
  }, [texturesPath, enableOrbitControls]);

  return (
    <div
      ref={containerRef}
      style={{
        width,
        height,
        backgroundColor: "black",
        overflow: "hidden",
        cursor: enableOrbitControls ? "grab" : "default",
      }}
    />
  );
};

export default EarthSatelliteScene;

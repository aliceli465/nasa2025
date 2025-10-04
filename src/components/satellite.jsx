import * as THREE from "three";

// Function to create a satellite geometry
export const createSatellite = () => {
  const satelliteGroup = new THREE.Group();

  // Main satellite body (50% larger)
  const bodyGeometry = new THREE.BoxGeometry(0.12, 0.18, 0.06);
  const bodyMaterial = new THREE.MeshBasicMaterial({ color: "#C0C0C0" });
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
  satelliteGroup.add(body);

  // Solar panels (50% larger)
  const panelGeometry = new THREE.BoxGeometry(0.015, 0.27, 0.03);
  const panelMaterial = new THREE.MeshBasicMaterial({ color: "#1A1A1A" });

  const leftPanel = new THREE.Mesh(panelGeometry, panelMaterial);
  leftPanel.position.set(-0.09, 0, 0);
  satelliteGroup.add(leftPanel);

  const rightPanel = new THREE.Mesh(panelGeometry, panelMaterial);
  rightPanel.position.set(0.09, 0, 0);
  satelliteGroup.add(rightPanel);

  // Antenna (50% larger)
  const antennaGeometry = new THREE.ConeGeometry(0.012, 0.09, 8);
  const antennaMaterial = new THREE.MeshBasicMaterial({ color: "#FFD700" });
  const antenna = new THREE.Mesh(antennaGeometry, antennaMaterial);
  antenna.position.set(0, 0.12, 0);
  satelliteGroup.add(antenna);

  return satelliteGroup;
};

export default createSatellite;

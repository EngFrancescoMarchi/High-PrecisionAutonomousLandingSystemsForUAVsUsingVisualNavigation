# Simulation Folder

This folder contains files and configurations for simulating the PX4 drone environment using Gazebo, specifically tailored for high-precision autonomous landing systems for UAVs using visual navigation.

## Contents

- **ComeUsare.yaml**: A YAML file containing terminal commands and instructions for setting up and running the simulation.
- **README.md**: This documentation file.
- **gz_x500_vision/**: Directory containing the model files for the x500 vision drone.
  - `model.config`: Configuration file for the drone model.
  - `model.sdf`: SDF (Simulation Description Format) file defining the drone's physical properties and sensors.
- **moving_platform/**: Directory containing the model for a moving platform used in the simulation.
  - `gva_platform.config`: Configuration file for the moving platform.
  - `gva_platform.sdf`: SDF file defining the moving platform's structure and behavior.
- **WorldSettings/**: Directory with world configuration files.
  - `default.sdf`: Default world settings for the Gazebo simulation environment.

## Functionality

These files have been modified to ensure a better simulation of the world or to implement sensors such as the camera where they were missing. The moving platform has been created 1 meter tall and is designed to simulate a dynamic landing target for the UAV.

The simulation integrates PX4 firmware with Gazebo to provide a realistic environment for testing autonomous landing algorithms, including visual navigation using cameras and ArUco markers.

## Usage Instructions

The following instructions are based on the commands provided in `ComeUsare.yaml`. Ensure you have PX4-Autopilot installed and configured properly.

### System Updates

```bash
sudo apt update && sudo apt upgrade -y
# Install recommended Nvidia drivers (essential for Gazebo)
sudo ubuntu-drivers autoinstall
```

### Python Environment Setup

```bash
sudo apt install python3-venv -y
cd ~
python3 -m venv tesi_env
# Activate the environment (do this every time you work on the project)
source tesi_env/bin/activate

# Install necessary libraries in the protected environment
pip install mavsdk opencv-contrib-python numpy
```

### Running the Drone Simulation in Gazebo

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500_vision
```

For windy conditions:

```bash
PX4_GZ_WORLD=forest make px4_sitl gz_x500_vision
```

### QGroundControl

To monitor and control the drone:

```bash
./QGroundControl.AppImage
```

### Creating the Target (ArUco Marker)

```bash
gz service -s /world/default/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 1000 --req 'sdf_filename: "/home/france/Desktop/aruco_marker.sdf", name: "aruco_target", pose: {position: {x: 4.5, y: 4.0, z: 0.01}, orientation: {x: 0.707, y: 0, z: 0, w: 0.707}}'
```

### Running the Landing Controller

```bash
cd ~/High-PrecisionAutonomousLandingSystemsForUAVsUsingVisualNavigation
python3 landing_controller.py
```

### Modifying the Drone Model

To edit the drone model:

```bash
gedit ~/PX4-Autopilot/Tools/simulation/gz/models/x500_vision/model.sdf
```

### Creating the Moving Platform

```bash
gz service -s /world/default/create \
--reqtype gz.msgs.EntityFactory \
--reptype gz.msgs.Boolean \
--req 'sdf_filename: "/home/france/PX4-Autopilot/Tools/simulation/gz/models/moving_platform/gva_platform.sdf", name: "gva_platform", pose: {position: {x: 3.5, y: -4.0, z: 0}}'
```

### Laboratory Setup

For laboratory testing:

```bash
gz service -s /world/default/create \
--reqtype gz.msgs.EntityFactory \
--reptype gz.msgs.Boolean \
--req 'sdf_filename: "/home/marchi/High-PrecisionAutonomousLandingSystemsForUAVsUsingVisualNavigation/Simulation/moving_platform/gva_platform.sdf", name: "gva_platform", pose: {position: {x: 0, y: 0, z: 0}}'

# Control the platform movement
gz topic -t /model/gva_platform/cmd_vel -m gz.msgs.Twist -p "linear: {x: 0.28, y: 0.28, z: 0.0}"
gz topic -t /model/gva_platform/cmd_vel -m gz.msgs.Twist -p "linear: {x: -0.14, y: 0.14, z: 0.0}"
gz topic -t /model/gva_platform/cmd_vel -m gz.msgs.Twist -p "linear: {x: 0, y: 0, z: 0.0}"
```

### Vision Testing via SSH

For testing vision on a remote machine:

```bash
ssh lazarus@192.168.10.224
nano test_visione.py
python3 test_visione.py
scp lazarus@192.168.10.224:~/log_visione.mp4 ~/Desktop
```

## Notes

- Adjust paths in the commands according to your system setup.
- Ensure all dependencies are installed before running the simulation.
- The simulation is designed for testing autonomous landing with visual feedback.
# Thesis
# High-Precision Autonomous Landing System for UAVs

This project implements a high-precision autonomous landing system for drones, based on computer vision (ArUco markers) and MAVLink protocol. The system is designed to operate on a Companion Computer (NVIDIA Jetson Nano) interfaced with a flight controller (Pixhawk) via serial telemetry and MAVSDK.

## 🏗 System Architecture

The system is divided into two operational environments: simulation (Gazebo) and physical hardware.

### Hardware Components
* **Flight Controller:** Pixhawk (PX4 Firmware)
* **Companion Computer:** NVIDIA Jetson Nano
* **Visual Sensor:** Standard USB camera (calibrated to 640x480 @ 30fps to optimize CPU load)
* **Ground Control Station (GCS):** Remote PC with QGroundControl on Linux OS.

### Network and Routing (MAVLink)
Communications between the Flight Controller, Companion Computer, and GCS are managed by `mavlink-router`, configured to route packets from the serial port (`/dev/ttyTHS1`) to two UDP endpoints:
* `127.0.0.1:14540` -> Local port for flight commands sent via MAVSDK in Python.
* `IP_GCS:14550` -> External port for telemetry to QGroundControl.

---

## 📂 Code Structure

The repository contains three main files, each with a specific purpose for drone development and testing:

### 1. `test_visione.py` (Hardware Validation)
Hardware testing script. Reads the video stream directly from the USB camera connected to the Jetson (`cv2.VideoCapture(0)`) and uses OpenCV (v4.7+) to detect ArUco markers (Dictionary `DICT_4X4_50`).
* **Headless Mode:** Designed to run via SSH, does not render graphical windows (`cv2.imshow` disabled) to avoid X11 server crashes and save resources.
* Returns real-time FPS to the terminal and pixel coordinates (`cx`, `cy`) of the error relative to the target center.

### 2. `vision_bridge.py` (Simulation Debugging)
Test node designed for the simulated environment.
* Subscribes to the `/camera` topic in Gazebo via `gz.transport13`.
* Converts image byte streams to NumPy arrays compatible with OpenCV (from RGB to BGR and grayscale).
* Calculates visual error and draws vectors and centers on screen for immediate visual feedback during simulations.

### 3. `landing_controller.py` (Mission Core)
The brain of the landing system. This script combines computer vision with flight control via MAVSDK.
* **Kalman Filter:** Estimates and cleans position measurements in pixel coordinates, mitigating frame losses or visual disturbances. Incorporates "Zero-Order Hold" (ZOH) logic to handle lost frames.
* **Parallax Correction:** Calculates the camera offset relative to the drone's center of mass to avoid misaligned landings.
* **PID Controller:** Generates `cmd_x` and `cmd_y` velocities based on visual error, with dynamic gains (Gain Scheduling) that become more conservative as altitude decreases. Includes Anti-Windup logic for the integral action.
* **Search Mode:** If the target is visually lost for more than 1.5 seconds, the drone initiates a spiral maneuver and climbs to a safe altitude (`SEARCH_CEILING`) to attempt visual reacquisition of the marker.

---

## 🚀 Startup Guide (Physical Test)

1. **Network Setup:** Ensure GCS and Jetson are on the same LAN/Hotspot.
2. **MAVLink Routing:** On the Jetson, start the router to open the ports:
   ```bash
   mavlink-routerd -e IP_GCS:14550 -e 127.0.0.1:14540 /dev/ttyTHS1:1000000
   ```

Finished simulation:
1. Control and Estimation Architecture (The "Brain")  
We moved from a lazy 25 Hz control loop to a multi-rate 100 Hz configuration.  
Control Frequency: The main loop now runs at 100 Hz, sending smooth setpoints to Pixhawk to zero motor response latency.  
Multi-Rate Kalman Filter: The filter runs the Predict phase at 100 Hz to keep the estimate smooth, while the Update phase occurs only when the camera provides a new frame (~30 Hz).  
Final Stability: We removed destructive oscillations at low altitude by replacing your old "power step" with linear Gain Scheduling that gradually reduces PID aggressiveness as the drone approaches the ground.  
Deadband: Introduced a 20-pixel dead zone below 60 cm to prevent the drone from chasing visual noise in the final moments of flight.  

2. Computer Vision and "Anti-Blur" Optimization (The "Eyes")  
We stopped treating the Nexigo webcam as a selfie camera and made it a serious navigation sensor.  
Projective Center: The error calculation no longer uses a trivial average, but the intersection of the ArUco marker diagonals for millimeter precision even with extreme angles.  
Manual Shutter: To eliminate motion blur (the blur that caused target loss in fast movements), we disabled auto-exposure and locked the shutter speed to fast values via OpenCV/v4l2.  
FPS Validation: Hardware tests confirmed that, despite the Jetson Nano load, vision maintains constant 30 FPS within a 100 Hz control loop.  

3. Connectivity and Hardware Configuration (The "Bridge")  
After various blind attempts, we finally established communication between the boards.  
Serial Link: We correctly identified the TELEM 2 port of the Pixhawk 6C as the onboard communication port.  
Baud Rate Alignment: We resolved the communication speed inconsistency, synchronizing Jetson and Pixhawk to 1,000,000 baud.  
PX4 Parameters: In QGroundControl, we set MAV_1_CONFIG to Telem 2 and MAV_1_MODE to Onboard, optimizing MAVLink traffic for the companion computer.  
Linux Permissions: We freed the /dev/ttyTHS1 port from system processes (like nvgetty) to allow your user to speak directly to the drone.

4. `REAL_landing.py` (Hardware-Validated Autonomous Landing)
The production-ready landing script that integrates all previous components into a fully autonomous landing system tested and validated on physical hardware.
* **Multi-Rate Control Loop:** Runs at 100 Hz for smooth command generation to the flight controller, while the vision subsystem operates independently at 30 FPS via a separate camera thread.
* **Advanced Kalman Filter:** Estimates drone position in pixel coordinates with Zero-Order Hold (ZOH) logic to gracefully handle frame drops and temporary target loss during fast descents.
* **Robust Parallax Correction:** Dynamically compensates for camera offset relative to the drone's center of mass, ensuring pixel-to-meter conversion accuracy at all altitudes.
* **Adaptive PID with Gain Scheduling:** Dynamically adjusts control gains based on altitude—aggressive far from ground, conservative near touchdown to prevent oscillations and overshooting.
* **Intelligent Search Mode:** When target is lost for >1.5 seconds, the drone executes a square spiral search pattern while climbing to regain visual lock without crashing.
* **Data Logging & Video Recording:** Continuous CSV logging of telemetry (position estimates, velocity commands, altitude, battery) and AVI video recording with periodic flushing to ensure data integrity even if the process crashes.
* **Battery Monitoring:** Real-time battery level tracking with automatic emergency descent (<20% remaining) to prevent in-flight power loss.
* **Production Robustness:** Implements crash-safe video recording (XVID codec with AVI format for append-friendly writes) and graceful error handling for hardware disconnections.
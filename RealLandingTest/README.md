# Real Landing Test Scripts

This folder contains two Python scripts for testing the hardware components of the high-precision autonomous landing system for UAVs using visual navigation.

## Files Overview

### 1. `test_visione.py` (Vision Hardware Test)

This script tests the computer vision pipeline using a physical USB camera connected to the NVIDIA Jetson Nano companion computer. It simulates the vision processing that occurs in the main landing controller.

**Key Features:**
- **Camera Initialization:** Opens the USB camera at 640x480 resolution and 30 FPS, with manual exposure settings to prevent motion blur.
- **ArUco Marker Detection:** Uses OpenCV to detect ArUco markers (dictionary DICT_4X4_50), prioritizing marker ID 4, then ID 0.
- **Projective Center Calculation:** Computes the true center of the detected marker using diagonal intersection for high precision, even at extreme angles.
- **Multithreaded Design:** Runs vision processing in a separate thread while simulating a 100 Hz control loop in the main thread.
- **Shared Buffer:** Uses a thread-safe buffer to pass measurements from vision thread to control thread, mimicking the real landing controller.
- **Video Recording:** Saves the processed video feed to `log_visione.mp4` for post-test analysis.
- **Headless Operation:** Designed for SSH execution without graphical output to avoid X11 crashes and save resources.
- **Real-time Output:** Prints target lock status, pixel errors (cx, cy), and frame reception rate every 10 control cycles.

**Usage:**
Run via SSH on the Jetson Nano:
```bash
python3 test_visione.py
```
The script will start video recording and display real-time status. Press Ctrl+C to stop.

**Purpose:** Validates camera functionality, marker detection accuracy, and multithreaded performance before real flight tests.

### 2. `test_seriale.py` (Serial Communication Test)

This script tests the serial communication link between the NVIDIA Jetson Nano companion computer and the Pixhawk flight controller using MAVSDK.

**Key Features:**
- **Serial Connection:** Attempts to connect to Pixhawk via `/dev/ttyTHS1` at 1,000,000 baud rate.
- **Connection Verification:** Waits for successful connection and telemetry establishment.
- **Attitude Reading:** Reads and displays the drone's attitude (roll, pitch, yaw) to confirm data flow.
- **Simple Validation:** Provides immediate feedback on whether the MAVLink communication is working.

**Usage:**
Run on the Jetson Nano:
```bash
python3 test_seriale.py
```
The script will attempt connection and display attitude data once connected. Press Ctrl+C to exit.

**Purpose:** Ensures proper serial communication setup between companion computer and flight controller before integrating with the full landing system.

## Dependencies

Both scripts require:
- Python 3.x
- OpenCV (for `test_visione.py`)
- MAVSDK (for `test_seriale.py`)

Install via:
```bash
pip install opencv-contrib-python mavsdk
```

## Hardware Requirements

- **NVIDIA Jetson Nano** with USB camera connected
- **Pixhawk flight controller** connected via serial (for `test_seriale.py`)
- Proper MAVLink routing configured (see main README)

## Notes

- These are standalone test scripts for hardware validation.
- `test_visione.py` operates in headless mode for remote execution.
- Ensure camera permissions and serial port access are configured correctly.
- Video output from `test_visione.py` is saved locally for analysis.
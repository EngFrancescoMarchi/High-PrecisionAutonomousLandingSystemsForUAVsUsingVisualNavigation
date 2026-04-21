import cv2
import numpy as np
import time
import threading

# Camera Settings
CAM_W, CAM_H = 640, 480
CENTER_X, CENTER_Y = CAM_W // 2, CAM_H // 2
CONTROL_FREQ = 100.0  # Simulate reading at 100Hz from the main loop
DT = 1.0 / CONTROL_FREQ

# --- Shared Buffer (Like in main) ---
class SharedBuffer:
    def __init__(self):
        self.measurement = None
        self.new_data = False
        self.last_receive_time = 0.0
        self.lock = threading.Lock() # Added lock for safety with real multithreading

    def write(self, u, v):
        with self.lock:
            if u is not None and v is not None:
                self.measurement = np.array([[u], [v]])
            else:
                self.measurement = None
            self.new_data = True
            self.last_receive_time = time.time()

    def read(self):
        with self.lock:
            data = self.measurement
            is_fresh = self.new_data
            self.new_data = False
            return data, is_fresh

shared_buffer = SharedBuffer()
stop_thread = False

# --- Camera Thread ---
# --- Camera Thread ---
def vision_thread_func():
    global stop_thread
    print("Initializing physical USB camera in HEADLESS mode...")
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 100)
    
    if not cap.isOpened():
        print("CRITICAL ERROR: Unable to open camera. USB cable connected?")
        stop_thread = True
        return

    # --- Video Recording Setup ---
    # Create the writer to save in MP4 format at 30 FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('log_visione.mp4', fourcc, 30.0, (CAM_W, CAM_H))
    print("Video recording started: saving to 'log_visione.mp4'")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        cx, cy = None, None

        if ids is not None:
            # Draw red/green borders around the ArUco on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            ids_list = ids.flatten().tolist()
            idx = -1
            
            if 4 in ids_list:
                idx = ids_list.index(4)
            elif 0 in ids_list:
                idx = ids_list.index(0)

            if idx != -1:
                c = corners[idx][0]
                
                # --- True Projective Center Calculation ---
                p0, p1, p2, p3 = c[0], c[1], c[2], c[3]
                A1, B1 = p2[1] - p0[1], p0[0] - p2[0]
                C1 = A1 * p0[0] + B1 * p0[1]
                A2, B2 = p3[1] - p1[1], p1[0] - p3[0]
                C2 = A2 * p1[0] + B2 * p1[1]
                
                det = A1 * B2 - A2 * B1
                if det != 0:
                    true_cx = (B2 * C1 - B1 * C2) / det
                    true_cy = (A1 * C2 - A2 * C1) / det
                else:
                    true_cx, true_cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                
                cx = int(true_cx) - CENTER_X
                cy = int(true_cy) - CENTER_Y
                
                # Draw a blue dot exactly at the calculated center
                cv2.circle(frame, (int(true_cx), int(true_cy)), 5, (255, 0, 0), -1)

        # Write data to the buffer for the control loop
        shared_buffer.write(cx, cy)

        # --- Video Output ---
        # Option 1: Save the frame to video file (safe via SSH)
        out.write(frame)
        
        # Option 2: Live Visualization. 
        # UNCOMMENT these two lines ONLY if you have a monitor connected directly to the drone
        # cv2.imshow("Live Vision Feedback", frame)
        # cv2.waitKey(1)

    # --- Final Cleanup ---
    cap.release()
    out.release() # Close the MP4 file to make it readable
    cv2.destroyAllWindows()
    print("Vision thread terminated. Video saved.")

# --- Main Loop (Simulates the control loop) ---
def main():
    global stop_thread
    
    # Start the camera thread
    vision_thread = threading.Thread(target=vision_thread_func)
    vision_thread.start()

    print(f"--- MULTITHREAD VISION TEST STARTED ({CAM_W}x{CAM_H}) ---")
    print(f"Simulated reading at {CONTROL_FREQ} Hz. Press Ctrl+C to exit.")
    
    time.sleep(2) # Wait for the camera to turn on

    if stop_thread:
        return

    next_wake_time = time.time() + DT
    loops = 0
    read_count = 0

    try:
        while True:
            # Read from the buffer exactly like Kalman does in the flight code
            measurement, is_new = shared_buffer.read()
            
            if is_new:
                read_count += 1 # Count how many fresh frames we are actually receiving
            
            # Print to screen only every 10 cycles (5 times per second) to not clog the SSH terminal
            loops += 1
            if loops % 10 == 0:
                if measurement is not None:
                    print(f"[Loop 50Hz] TARGET LOCKED | cx: {int(measurement[0][0]):4d}, cy: {int(measurement[1][0]):4d} | New Frames seen: {read_count}/10")
                else:
                    print(f"[Loop 50Hz] TARGET LOST   | No data in buffer          | New Frames seen: {read_count}/10")
                read_count = 0 # Reset the fresh reads counter

            # Synchronization at 50Hz
            sleep_time = next_wake_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_wake_time += DT

    except KeyboardInterrupt:
        print("\nTest interrupted manually by user.")
    finally:
        stop_thread = True
        vision_thread.join()
        print("Clean shutdown completed.")

if __name__ == "__main__":
    main()
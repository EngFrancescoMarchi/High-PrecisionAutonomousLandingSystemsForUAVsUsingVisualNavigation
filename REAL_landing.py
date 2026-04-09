import sys
import cv2
import numpy as np
import time
import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed) 
import threading
import csv          # <-- AGGIUNTO: per salvare CSV senza pandas/matplotlib
import os           # <-- AGGIUNTO: per path management
 
FREQ = 100.0             # Updating at 100Hz for better performance with HD stream
DT = 1.0 / FREQ        
TARGET_ALTITUDE = 5.5  # Target altitude for initial hover before descent (meters)
ALIGN_THRESHOLD = 110    # Pixel tolerance to start descent
 
# Camera Parameters (gz_x500_vision standard + HD)
CAM_W, CAM_H = 640,480
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
 
# ===========================================================================
#   FIX 1 - CSV SAVE FUNCTION (no matplotlib, no pandas, no crash risk)
# ===========================================================================
def save_csv(log_data, filename=None):
    """
    Salva log_data su CSV usando solo la libreria standard.
    Nessun import di matplotlib/pandas => nessun rischio di crash.
    """
    if filename is None:
        filename = f"log_volo_{int(time.time())}.csv"
    
    keys = list(log_data.keys())
    n_rows = len(log_data[keys[0]])
    
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)                       # header
            for i in range(n_rows):
                row = [log_data[k][i] for k in keys]
                writer.writerow(row)
        print(f"[CSV] Salvato: {filename} ({n_rows} righe)")
    except Exception as e:
        print(f"[CSV] ERRORE nel salvataggio: {e}")
 
# --- Zero-Order Hold Buffer ---
class SharedBuffer:
    def __init__(self):
        self.measurement = None 
        self.frame = None
        self.new_data = False
        self.last_receive_time = 0.0
        self.lock = threading.Lock()
 
    def write(self, u, v, frame):
        with self.lock:
            if u is not None and v is not None:
                self.measurement = np.array([[u], [v]])
            else:
                self.measurement = None
            self.frame = frame
            self.new_data = True
            self.last_receive_time = time.time()
 
    def read(self):
        with self.lock:
            data = self.measurement
            is_fresh = self.new_data
            frame = self.frame
            self.new_data = False
            return data, frame, is_fresh
 
# --- Kalman Filter ---
class LandingKalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((4, 1))
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = np.eye(4) * 0.02
        self.R = np.eye(2) * 15.0  
        self.P = np.eye(4) * 1.0
 
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
 
    def update(self, z):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - (K @ self.H)) @ self.P
 
# --- Vision Callback ---
shared_buffer = SharedBuffer()
 
# ===========================================================================
#   FIX 2 - CAMERA THREAD con video recording robusto
# ===========================================================================
class CameraThread(threading.Thread):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self.daemon = False
        self.running = True
 
    def run(self):
        print("[Vision] Physical camera initialization...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, FREQ)
        
        # Camera settings for outdoor use (adjust based on your camera)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Exposure value, adjust for brightness
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
        cap.set(cv2.CAP_PROP_CONTRAST, 128)
        cap.set(cv2.CAP_PROP_SATURATION, 128)
 
        if not cap.isOpened():
            print("[Vision] CRITICAL ERROR: Camera not found!")
            self.running = False
            return
 
        # -----------------------------------------------------------------
        #  FIX VIDEO: Uso XVID/AVI invece di mp4v/mp4.
        #  AVI e' un formato "append-friendly": ogni frame scritto va 
        #  direttamente su disco. MP4 richiede il release() finale per 
        #  scrivere l'header, e se il processo crasha perdi tutto.
        # -----------------------------------------------------------------
        VIDEO_FPS = 30.0   # FPS reali della camera USB (non FREQ=100)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_filename = f"log_landing_{int(time.time())}.avi"
        out = cv2.VideoWriter(video_filename, fourcc, VIDEO_FPS, (CAM_W, CAM_H))
        
        if not out.isOpened():
            print(f"[Vision] WARNING: VideoWriter failed con XVID, provo MJPG...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_filename = f"log_landing_{int(time.time())}.avi"
            out = cv2.VideoWriter(video_filename, fourcc, VIDEO_FPS, (CAM_W, CAM_H))
 
        print(f"[Vision] USB Webcam Ready: {CAM_W}x{CAM_H}")
        print(f"[Vision] Video recording: {video_filename}")
        
        frame_count = 0
        FLUSH_INTERVAL = int(VIDEO_FPS * 3)  # Flush ogni ~3 secondi
        
        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            frame_display = frame.copy()
            gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            cx, cy = None, None
 
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame_display, corners, ids)
                ids_list = ids.flatten().tolist()
                if 4 in ids_list: idx = ids_list.index(4)
                elif 0 in ids_list: idx = ids_list.index(0)
                else: idx = -1
 
                if idx != -1:
                    c = corners[idx][0]
                    dyn_center_x = CAM_W // 2
                    dyn_center_y = CAM_H // 2
                    
                    # Calculation of the intersection point of the diagonals for better centering
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
                    
                    cx = int(true_cx) - dyn_center_x
                    cy = int(true_cy) - dyn_center_y
                    
                    cv2.circle(frame_display, (dyn_center_x + cx, dyn_center_y + cy), 5, (0, 0, 255), -1)
 
            # Write video frame
            out.write(frame_display)
            self.buffer.write(cx, cy, frame_display)
            frame_count += 1
            
            # -----------------------------------------------------------------
            #  FIX VIDEO: Flush periodico - riapre il VideoWriter ogni ~3 sec.
            #  In caso di crash, perdi al massimo 3 secondi di video.
            # -----------------------------------------------------------------
            if frame_count % FLUSH_INTERVAL == 0 and frame_count > 0:
                out.release()
                #out = cv2.VideoWriter(video_filename, fourcc, VIDEO_FPS, (CAM_W, CAM_H))
                # Nota: questo sovrascrive il file, quindi il video contiene
                # solo gli ultimi ~3 sec. Se vuoi mantenere tutto, usa segmenti:
                video_filename = f"log_landing_seg{frame_count}.avi"
                out = cv2.VideoWriter(video_filename, fourcc, VIDEO_FPS, (CAM_W, CAM_H))
        
        cap.release()
        out.release()
        print(f"[Vision] Camera spenta. Frame totali registrati: {frame_count}")
 
    def stop(self):
        self.running = False
 
# --- Telemetry Background Task ---
current_alt = 0.0
current_battery = 100.0  # Initialize battery percentage
low_battery = False
async def telemetry_loop(drone):
    global current_alt, current_battery, low_battery
    async for pos in drone.telemetry.position():
        current_alt = pos.relative_altitude_m
    async for battery in drone.telemetry.battery():
        current_battery = battery.remaining_percent * 100  # Convert to percentage
        if current_battery < 20.0 and not low_battery:
            print(f"WARNING: Low battery ({current_battery:.1f}%) - Initiating emergency landing!")
            low_battery = True
 
log_data = {}
cam_thread = None
 
# --- Main Loop ---
async def run():
    global current_alt, log_data, cam_thread
    drone = System()
    print("Connessione al drone su /dev/ttyTHS1 (1M baud)...")
    await drone.connect(system_address="serial:///dev/ttyTHS1:1000000")
    
    print("In attesa della connessione MAVLink...")
    async for state in drone.core.connection_state():
        if state.is_connected: 
            print("--- HARDWARE LINK ATTIVO ---")
            break
    async for state in drone.core.connection_state():
        if state.is_connected: break
    kf = LandingKalmanFilter(DT)
    asyncio.create_task(telemetry_loop(drone))
    cam_thread = CameraThread(shared_buffer)
    cam_thread.start()
    
    # Wait for initial telemetry data
    await asyncio.sleep(2)
    # PID Gains (100Hz + HD)
    KP_X, KD_X = 0.0012, 0.003
    KP_Y, KD_Y = 0.0012, 0.003
    KI = 0.0005   # Integral Gain 
    
    cruise_altitude_reached = False 
    
    # --- Search Mode to Find Target ---
    search_active = False
    last_seen_time = time.time()
    search_start_time = 0
    search_leg_index = 0
    search_leg_duration = 2.0 
    base_search_speed = 1.2  
    
    # --- Integral Terms ---
    integ_x = 0.0
    integ_y = 0.0
    integ_max = 1000.0 # Anti-Windup Limit
 
    # Arming Sequence
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_local_position_ok and health.is_home_position_ok:
            print("-- Check unlocked, global/local position and home position: OK")
        break
    print("-- Arming & Takeoff")
    try:
        await drone.action.arm()
    except Exception as e:
        print(f"!!! Critical error: {e} !!!")
        cam_thread.stop()
        return
    await asyncio.sleep(8)
    
    print(f"--- MISSION START ({FREQ} Hz) ---")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0,0,0,0))
    try: await drone.offboard.start()
    except OffboardError: return
 
    next_wake_time = time.time() + DT
    # --- Data Logging Setup ---
    log_data = {
        'time': [],
        'alt': [],
        'pos_x_est': [],
        'pos_y_est': [],
        'vel_x_cmd': [],  
        'vel_y_cmd': [],
        'target_visible': [],
        'battery': []
    }
    start_log_time = time.time()
    while True:
    
        measurement, frame_to_show, is_new = shared_buffer.read()
    
        # Kalman Prediction & Update
        est_state = kf.predict() 
        if is_new and measurement is not None:
            kf.update(measurement)
            last_seen_time = time.time()
        
        est_x, est_vx = est_state[0][0], est_state[1][0]
        est_y, est_vy = est_state[2][0], est_state[3][0]
        
        # --- Parallax Correction ---
        CAMERA_OFFSET_Y = 0.05   # Camera offset backward from Center of Mass 
        CAMERA_OFFSET_Z = 0.15   # Camera offset downward from Center of Mass 
        FOCAL_LENGTH    = 550.0 # Focal length in pixels (for 720p/1080p; for 640x480 use ~550)
 
        # 1. Calculation of effective altitude for parallax correction
        cam_alt = max(current_alt - CAMERA_OFFSET_Z, 1) 
        
        # 2. Offset pixel to meter conversion (parallax)
        expected_pixel_offset = (CAMERA_OFFSET_Y * FOCAL_LENGTH) / cam_alt
        
        # 3. Application of correction
        est_x = est_x
        est_y = est_y + expected_pixel_offset
        
        # --- Control Section ---
        cmd_x, cmd_y, cmd_z = 0.0, 0.0, 0.0
        
        # Emergency check for low battery
        if low_battery:
            cmd_z = 0.5  # Emergency descent
        else:
        
            # State 1: Takeoff
            if not cruise_altitude_reached:
                if current_alt >= TARGET_ALTITUDE - 0.5:
                    print("--- ALTITUDE REACHED ---")
                    cruise_altitude_reached = True
                    last_seen_time = time.time() # Reset sight timer
                else:
                    cmd_z = -1.0
                    if measurement is not None: # Preventing centering on noise during takeoff
                        cmd_y = (est_x * KP_X)
                        cmd_x = -((est_y * KP_Y))
 
            # State 2: Descent and Search
            else:
                # Check if we have a recent sighting of the target (within the last 1.5 seconds)
                target_visible = (time.time() - last_seen_time) < 1.5
 
                # --- 2A. Target Tracking ---
                if target_visible:
                    # Reset Search Mode
                    if search_active:
                        print(">>> TARGET LOCKED! STOP RESEARCH <<<")
                        search_active = False
                        search_leg_index = 0
                        # Reset Integral on finding to avoid jerks
                        integ_x, integ_y = 0.0, 0.0
                # Damper scales the calculated force; here we use 40% to avoid shaking
                    # Gain Scheduling
                    dampener = np.clip((current_alt - 0.5) / 1.2, 0.4, 1.0)
                    max_speed_xy = np.clip(current_alt * 0.8, 0.1, 1.4)
                    MAX_FF = 0.0035
                    err_dist = np.hypot(est_x, est_y)
                    spatial_multiplier = np.clip((err_dist - 50.0) / 85.0, 0.0, 1.0)
                    spatial_ff_gain = MAX_FF * spatial_multiplier
                    # --- COMPLETE PID CALCULATION (P + I + D + FF) ---
                    cone_multiplier = np.interp(current_alt, [0.7, 2.0, TARGET_ALTITUDE], [1, 1.25, 1.5])
                    current_align_thresh = ALIGN_THRESHOLD * cone_multiplier
                    is_aligned = (abs(est_x) < current_align_thresh and abs(est_y) < current_align_thresh)
                    # --- Freeze Integral in Last Meter ---
                    i_dampener = np.clip((current_alt - 1.0) / 2.0, 0.3, 1.0)
                    if current_alt < 0.8:
                        i_dampener = 0.0
                    integ_x += (est_x * DT) * i_dampener
                    integ_y += (est_y * DT) * i_dampener
                        # Standard Anti-Windup
                    integ_x = np.clip(integ_x, -integ_max, integ_max)
                    integ_y = np.clip(integ_y, -integ_max, integ_max)
                    # 4. Total PID Calculation
                    # Y Axis (Roll)
                    cmd_y = (est_x * KP_X * dampener) + \
                            (est_vx * KD_X * dampener) + \
                            (integ_x * KI) + \
                            (est_vx * spatial_ff_gain)
                    
                    # X Axis (Pitch)
                    cmd_x = -((est_y * KP_Y * dampener) + \
                            (est_vy * KD_Y * dampener) + \
                            (integ_y * KI) + \
                            (est_vy * spatial_ff_gain))
                    
                    # --- End PID Calculation ---
 
                    # Clamping
                    cmd_x = np.clip(cmd_x, -max_speed_xy, max_speed_xy)
                    cmd_y = np.clip(cmd_y, -max_speed_xy, max_speed_xy)
 
                    # --- Logging (Inside the Loop) ---
                    current_log_time = time.time() - start_log_time
                    log_data['time'].append(current_log_time)
                    log_data['alt'].append(current_alt)
                    log_data['pos_x_est'].append(est_x)
                    log_data['pos_y_est'].append(est_y)
                    log_data['vel_x_cmd'].append(cmd_x)
                    log_data['vel_y_cmd'].append(cmd_y)
                    log_data['target_visible'].append(1 if target_visible else 0)
                    log_data['battery'].append(current_battery)
 
                    if is_aligned:
                        cmd_z = np.interp(current_alt, [0.35, 1.5], [0.45, 0.8])
                    else:
                        # Corrective hovering
                        cmd_z = 0.0 
 
                # --- 2B. Target Lost: Recognition ---
                else:
                    # 1. Reset Integral Terms
                    integ_x, integ_y = 0.0, 0.0
                    
                    time_since_loss = time.time() - last_seen_time
                    
                    # Phase 1: Wait (Anti-Glitch) - 3 seconds
                    if time_since_loss < 3:
                        cmd_x, cmd_y, cmd_z = 0.0, 0.0, 0.0
                        if time_since_loss > 1.5: # Print only after 1 second to not spam
                            print(f"WAITING... {time_since_loss:.2f}")
                    # Phase 2: Search Mode (Spiral + Ascent)
                    else:
                        if not search_active:
                            print(f">>> LOST IN LANDING, INITIATING SEARCH <<<")
                            search_active = True
                            search_start_time = time.time()
                            search_leg_index = 0
                            search_leg_duration = 1.5 # Fast spiral
                        
                        dt_search = time.time() - search_start_time
 
                        # Spiral Search Management
                        if dt_search > search_leg_duration:
                            search_leg_index += 1
                            search_start_time = time.time()
                            if search_leg_index % 2 == 0:
                                search_leg_duration += 1.0 
 
                        direction = search_leg_index % 4
                        spd = base_search_speed
                        
                        if direction == 0:   cmd_x, cmd_y = spd, 0.0
                        elif direction == 1: cmd_x, cmd_y = 0.0, spd
                        elif direction == 2: cmd_x, cmd_y = -spd, 0.0
                        elif direction == 3: cmd_x, cmd_y = 0.0, -spd
                        
                        # --- Crucial Modification: Ascent During Search ---
                        SEARCH_CEILING = 5.0
                        
                        if current_alt < SEARCH_CEILING:
                            cmd_z = -1.0 # Go up to regain sight
                        else:
                            cmd_z = 0.0  # Maintain altitude if already high
 
        # --- Touchdown ---
        if current_alt < 0.2 and cruise_altitude_reached:
             print("--- TOUCHDOWN ---")
             await drone.offboard.set_velocity_body(VelocityBodyYawspeed(cmd_x, cmd_y, 0.5,0))
             try: await drone.offboard.stop()
             except: pass
             await drone.action.land()
             break
 
        # --- Send Commands ---
        if low_battery:
            cmd_x, cmd_y = 0.0, 0.0
            cmd_z = 0.5  # Force emergency descent
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(cmd_x, cmd_y, cmd_z, 0.0))
 
        # --- Timing Control ---
        sleep_time = next_wake_time - time.time()
        if sleep_time > 0: await asyncio.sleep(sleep_time)
        next_wake_time += DT
 
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        print("\n!!! Interrotto dall'utente !!!")
    except Exception as e:
        print(f"Errore imprevisto: {e}")
    finally:
        # ==================================================================
        #  FIX 1: Salva CSV in modo sicuro (no matplotlib, no crash)
        # ==================================================================
        if 'time' in log_data and len(log_data['time']) > 0:
            print(f"[SAVE] Salvataggio dati ({len(log_data['time'])} punti)...")
            save_csv(log_data)
        else:
            print("[SAVE] Nessun dato registrato da salvare.")
            
        print("Cleaning and closing...")        
        if cam_thread is not None:
            cam_thread.stop()
            cam_thread.join(timeout=5)  # Timeout per evitare hang infinito
        
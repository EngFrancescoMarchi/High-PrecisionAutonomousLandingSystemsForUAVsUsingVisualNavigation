import sys
import cv2
import numpy as np
import time
import asyncio
from mavsdk import System
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.pose_pb2 import Pose
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed, PositionNedYaw)
import matplotlib.pyplot as plt
import pandas as pd # Se vuoi salvare in Excel/CSV
from plot_results import plot_results

try:
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
except ImportError:
    print("ERRORE CRITICO")
    sys.exit(1)

FREQ = 25.0             #upadating at 30Hz for better performance with HD stream
DT = 1.0 / FREQ        
TARGET_ALTITUDE = 5.5   # Target altitude for initial hover before descent (meters)
ALIGN_THRESHOLD = 60    # Pixel tolerance to start descent

# Camera Params (gz_x500_vision standard + HD)
CAM_W, CAM_H = 640,480
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
# --- ZOH as buffer ---
class SharedBuffer:
    def __init__(self):
        self.measurement = None 
        self.frame = None
        self.new_data = False
        self.last_receive_time = 0.0

    def write(self, u, v, frame):
        if u is not None and v is not None:
            self.measurement = np.array([[u], [v]])
        else:
            self.measurement = None
        self.frame = frame
        self.new_data = True
        self.last_receive_time = time.time()

    def read(self):
        data = self.measurement
        is_fresh = self.new_data
        frame = self.frame
        self.new_data = False
        return data, frame, is_fresh

# --- KALMAN FILTER ---
class LandingKalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((4, 1))
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = np.eye(4) * 0.01 
        self.R = np.eye(2) * 30.0  
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

# --- VISION CALLBACK ---
shared_buffer = SharedBuffer()

def vision_callback(msg):
    try:
        img_buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = img_buf.reshape((msg.height, msg.width, 3))
        
        frame_display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = detector.detectMarkers(gray)
        cx, cy = None, None
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_display, corners, ids)
            ids_list = ids.flatten().tolist()
            if 4 in ids_list:
                idx = ids_list.index(4)
                
            elif 0 in ids_list:
                idx = ids_list.index(0)
                
            else:
                idx = -1
    
            if idx != -1:
                c = corners[idx][0]
                dyn_center_x = msg.width // 2
                dyn_center_y = msg.height // 2
                
                # --- CALCOLO VERO CENTRO PROIETTIVO (Intersezione Diagonali) ---
                p0, p1, p2, p3 = c[0], c[1], c[2], c[3]
                
                # Retta 1 (Diagonale da p0 a p2)
                A1 = p2[1] - p0[1]
                B1 = p0[0] - p2[0]
                C1 = A1 * p0[0] + B1 * p0[1]
                
                # Retta 2 (Diagonale da p1 a p3)
                A2 = p3[1] - p1[1]
                B2 = p1[0] - p3[0]
                C2 = A2 * p1[0] + B2 * p1[1]
                
                det = A1 * B2 - A2 * B1
                
                if det != 0:
                    true_cx = (B2 * C1 - B1 * C2) / det
                    true_cy = (A1 * C2 - A2 * C1) / det
                else:
                    # Fallback di emergenza
                    true_cx, true_cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                
                cx = int(true_cx) - dyn_center_x
                cy = int(true_cy) - dyn_center_y
                
                cv2.circle(frame_display, (dyn_center_x + cx, dyn_center_y + cy), 5, (0, 0, 255), -1)
        shared_buffer.write(cx, cy, frame_display) 
    except Exception:
        pass

# --- TELEMETRY BACKGROUND ---
current_alt = 0.0
async def telemetry_loop(drone):
    global current_alt
    async for pos in drone.telemetry.position():
        current_alt = pos.relative_altitude_m
log_data = {}
# --- MAIN LOOP ---
async def run():
    global current_alt, log_data
    # Setup
    drone = System()
    await drone.connect(system_address="udp://:14540")
    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected: break

    node = Node()
    node.subscribe(Image, "/camera", vision_callback)
    kf = LandingKalmanFilter(DT)
    asyncio.create_task(telemetry_loop(drone))

    # PID Gains (30Hz + HD)
    KP_X, KD_X = 0.0012, 0.003
    KP_Y, KD_Y = 0.0012, 0.003
    KI = 0.00035   # <--- NUOVO: Integrale Gain
    
    cruise_altitude_reached = False 
    
    # --- Loop over the place to find target (SEARCH MODE) ---
    search_active = False
    last_seen_time = time.time()
    search_start_time = 0
    search_leg_index = 0
    search_leg_duration = 2.0 
    base_search_speed = 1.2  
    
    # --- INT ---
    integ_x = 0.0
    integ_y = 0.0
    integ_max = 1000.0 # Anti-Windup Limit

    # Decollo
    print("-- Arming & Takeoff")
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(8)
    
    print(f"--- MISSION START ({FREQ} Hz) ---")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0,0,0,0))
    try: await drone.offboard.start()
    except OffboardError: return

    next_wake_time = time.time() + DT
    # --- DATA LOGGING SETUP ---
    log_data = {
        'time': [],
        'alt': [],
        'pos_x_est': [],  # Posizione stimata dal Kalman
        'pos_y_est': [],
        'vel_x_cmd': [],  # Comando inviato
        'vel_y_cmd': [],
        'target_visible': [] # 1 se visto, 0 se perso
    }
    start_log_time = time.time()
    while True:
    
        measurement, frame_to_show, is_new = shared_buffer.read()
        if frame_to_show is not None:
            cv2.imshow("Drone Gazebo Vision", frame_to_show)
            cv2.waitKey(1)
        # Kalman Prediction & Update
        est_state = kf.predict() 
        if is_new and measurement is not None:
            kf.update(measurement)
            last_seen_time = time.time()
        
        est_x, est_vx = est_state[0][0], est_state[1][0]
        est_y, est_vy = est_state[2][0], est_state[3][0]
        
        # --- Parallax Correction ---
        CAMERA_OFFSET_X = -0.05   # Camera forward of COM 
        CAMERA_OFFSET_Z = 0.15   # Camera lower than COM 
        FOCAL_LENGTH    = 550.0 # Pixel ( 720p/1080p. Se 640x480 usa ~550)

        # 1. Calculo Altitudine Effettiva della Camera
        # If the camera is below the COM, the effective altitude for parallax is above, it's lower.
        cam_alt = max(current_alt - CAMERA_OFFSET_Z, 0.4) 
        
        # 2. Offset pixel to meter conversion (parallax)
        expected_pixel_offset = (CAMERA_OFFSET_X * FOCAL_LENGTH) / cam_alt
        
        # 3. Application
        #If the camera is forward of the COM, the target appears shifted in the opposite direction of the movement, so we subtract the expected pixel offset from the estimated position to get a more accurate error for control.
        est_x = est_x + expected_pixel_offset
        est_y = est_y  # No correction needed on Y for forward offset
        
        # --- B. CONTROLLO ---
        cmd_x, cmd_y, cmd_z = 0.0, 0.0, 0.0
        
        # STATO 1: Takeoff
        if not cruise_altitude_reached:
            if current_alt >= TARGET_ALTITUDE - 0.5:
                print("--- QUOTA RAGGIUNTA ---")
                cruise_altitude_reached = True
                last_seen_time = time.time() # Reset timer vista
            else:
                cmd_z = -1.0
                if measurement is not None: # Centratura preventiva
                     cmd_y = (est_x * KP_X)
                     cmd_x = -((est_y * KP_Y))

        # STATO 2: descent + search
        else:
            # Check se abbiamo il target ORA
            target_visible = (time.time() - last_seen_time) < 1.5

            # --- 2A. Target Tracking ---
            if target_visible:
                # Reset Research
                if search_active:
                    print(">>> TARGET LOCKED! STOP RESEARCH <<<")
                    search_active = False
                    search_leg_index = 0
                    # Reset Integral on finding to avoid jerks
                    integ_x, integ_y = 0.0, 0.0
            #Damper is the scale of the calculated force, 
            # in this case we will use 40% of calculated, avoid shaking
                # Gain Scheduling
                if current_alt < 0.65:
                    dampener = 0.35
                    max_speed_xy = 0.4 
                else:
                    dampener = 1.0
                    max_speed_xy = 1.4

                # --- CALCOLO PID COMPLETO (P + I + D + FF) ---
                
                # --- Cutting Integral last meter (FREEZE LOGIC) ---
                INTEGRAL_CUTOFF_HEIGHT = 0.7
                
                if current_alt > INTEGRAL_CUTOFF_HEIGHT:
                    # FFlying above cutoff, integral is active
                    integ_x += est_x * DT
                    integ_y += est_y * DT
                    
                    # Anti-Windup standard
                    integ_x = np.clip(integ_x, -integ_max, integ_max)
                    integ_y = np.clip(integ_y, -integ_max, integ_max)
                else:
                    
                    pass
                # 3. Feed-Forward Gain (Stima Velocità)
                if abs(est_x) < 25 or abs(est_y) < 25:
                    ff_gain = 0.0  # Se siamo molto vicini, disabiliti
                elif current_alt < 1.15:
                    ff_gain = 0.0020  # Guadagno più conservativo in discesa
                else:
                    ff_gain = 0.0035 

                # 4. Total PID
                # Asse Y (Roll)
                cmd_y = (est_x * KP_X * dampener) + \
                        (est_vx * KD_X * dampener) + \
                        (integ_x * KI) + \
                        (est_vx * ff_gain)
                
                # Asse X (Pitch)
                cmd_x = -((est_y * KP_Y * dampener) + \
                          (est_vy * KD_Y * dampener) + \
                          (integ_y * KI) + \
                          (est_vy * (ff_gain)))
                
                # --- END PID ---
#In the landing zone we cannot assure all the pixel as before, so we will set a treshhold
                
                # Clamping
                cmd_x = np.clip(cmd_x, -max_speed_xy, max_speed_xy)
                cmd_y = np.clip(cmd_y, -max_speed_xy, max_speed_xy)

                # Gestione Discesa
                current_align_thresh = ALIGN_THRESHOLD if current_alt > 0.85 else (ALIGN_THRESHOLD * 2.5)
                is_aligned = (abs(est_x) < current_align_thresh and abs(est_y) < current_align_thresh)
                
                # --- LOGGING (Dentro il While) ---
                current_log_time = time.time() - start_log_time
                log_data['time'].append(current_log_time)
                log_data['alt'].append(current_alt)
                log_data['pos_x_est'].append(est_x) # O l'errore raw se preferisci
                log_data['pos_y_est'].append(est_y)
                log_data['vel_x_cmd'].append(cmd_x)
                log_data['vel_y_cmd'].append(cmd_y)
                log_data['target_visible'].append(1 if target_visible else 0)
                if is_aligned:
                    final_descent_speed = 0.13 if current_alt < 0.85 else 0.30
                    cmd_z = final_descent_speed
                else:
                    # Hovering correttivo
                    cmd_z = 0.0 

            # --- 2B. TARGET LOST:recognition ---
            else:
                # 1. Reset of I
                integ_x, integ_y = 0.0, 0.0
                
                time_since_loss = time.time() - last_seen_time
                
                # Fase 1: Wait (Anti-Glitch) - 1.5 secondi
                if time_since_loss < 2.5:
                    cmd_x, cmd_y, cmd_z = 0.0, 0.0, 0.0
                    if time_since_loss > 1.5: # Print solo dopo 1 secondo per non spammare
                        print(f"WAITING... {time_since_loss:.2f}")
                # Fase 2: Search Mode (Spirale + Risalita)
                else:
                    if not search_active:
                        print(f">>> LOST IN LANDING, INITIATING SEARCH <<<")
                        search_active = True
                        search_start_time = time.time()
                        search_leg_index = 0
                        search_leg_duration = 1.5 # Spirale veloce
                    
                    dt_search = time.time() - search_start_time
                    
                    # Gestione Spirale (invariata)
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
                    
                    # --- LA MODIFICA CRUCIALE: RISALITA ---
                    # If we lost target, we might be too low to see it again. To avoid getting stuck in a blind spot, we will command a slow ascent until we reach a certain ceiling where we can search effectively.
                    SEARCH_CEILING = 5.0
                    
                    if current_alt < SEARCH_CEILING:
                        cmd_z = -1.0 # Go up to regain sight
                    else:
                        cmd_z = 0.0  # Maintain altitude if already high

        # --- C. TOUCHDOWN ---
        if current_alt < 0.23 and cruise_altitude_reached:
             print("--- TOUCHDOWN ---")
             await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0,0,0,0))
             try: await drone.offboard.stop()
             except: pass
             await drone.action.kill()
             break

        # --- D. COMANDO ---
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(cmd_x, cmd_y, cmd_z, 0.0))

        # --- E. TIMING ---
        sleep_time = next_wake_time - time.time()
        if sleep_time > 0: await asyncio.sleep(sleep_time)
        next_wake_time += DT

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        print("\n!!! Interrotto da France (utente) !!!")
    except Exception as e:
        print(f"Errore imprevisto: {e}")
    finally:
        # Ora controlliamo se log_data ha effettivamente dei dati prima di plottare
        if 'time' in log_data and len(log_data['time']) > 0:
            print(f"Salvataggio dati ({len(log_data['time'])} punti)...")
            plot_results(log_data)
        else:
            print("Nessun dato registrato da plottare.")
            
        print("Pulizia e chiusura...")
        # Questo forza la chiusura dei thread appesi di MAVSDK/OpenCV
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
import pandas as pd # If you want to save in Excel/CSV
from plot_results import plot_results

try:
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
except ImportError:
    print("ERRORE CRITICO")
    sys.exit(1)

FREQ = 100.0             #
DT = 1.0 / FREQ        
TARGET_ALTITUDE = 5.0  # Target altitude for initial hover before descent (meters)
ALIGN_THRESHOLD = 110    # Pixel tolerance to start descent

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
                
                # --- TRUE PROJECTIVE CENTER CALCULATION (Diagonal Intersection) ---
                p0, p1, p2, p3 = c[0], c[1], c[2], c[3]
                
                # Line 1 (Diagonal from p0 to p2)
                A1 = p2[1] - p0[1]
                B1 = p0[0] - p2[0]
                C1 = A1 * p0[0] + B1 * p0[1]
                
                # Line 2 (Diagonal from p1 to p3)
                A2 = p3[1] - p1[1]
                B2 = p1[0] - p3[0]
                C2 = A2 * p1[0] + B2 * p1[1]
                
                det = A1 * B2 - A2 * B1
                
                if det != 0:
                    true_cx = (B2 * C1 - B1 * C2) / det
                    true_cy = (A1 * C2 - A2 * C1) / det
                else:
                    # Emergency fallback
                    true_cx, true_cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                
                cx = int(true_cx) - dyn_center_x
                cy = int(true_cy) - dyn_center_y
                
                cv2.circle(frame_display, (dyn_center_x + cx, dyn_center_y + cy), 5, (0, 0, 255), -1)
        shared_buffer.write(cx, cy, frame_display) 
    except Exception:
        pass
def calculate_flight_stats(log_data):
    # Convert to numpy arrays
    pos_x = np.array(log_data['pos_x_est'])
    pos_y = np.array(log_data['pos_y_est'])
    times = np.array(log_data['time'])
    alts  = np.array(log_data['alt'])

    # --- 1. PURE DESCENT TIME CALCULATION ---
    # Find the maximum altitude reached (the peak at the end of takeoff)
    peak_alt = np.max(alts)
    
    # Find all indices where the drone was near the peak (e.g., within 10 cm).
    # This helps us ignore the "hovering" phase at high altitude.
    hovering_indices = np.where(alts >= peak_alt - 0.10)[0]
    
    # The last of these indices is the exact moment when unstoppable descent begins
    idx_descent_start = hovering_indices[-1]
    
    descent_start_time = times[idx_descent_start]
    touchdown_time = times[-1]
    
    actual_descent_time = touchdown_time - descent_start_time

    # --- 2. FINAL ERRORS CALCULATION (TOUCHDOWN) ---
    final_err_x = pos_x[-1]
    final_err_y = pos_y[-1]

    # --- 3. PRINT TO SCREEN ---
    print("\n" + "="*40)
    print(" LANDING STATISTICS (PURE DESCENT) ")
    print("="*40)
    print(f"Descent start altitude: {alts[idx_descent_start]:.2f} m")
    print(f"Takeoff/hovering time:  {descent_start_time:.2f} s")
    print(f"PURE DESCENT TIME:       {actual_descent_time:.2f} s")
    print("-" * 40)
    print(f"Touchdown Err X:                {final_err_x:.2f} px")
    print(f"Touchdown Err Y:                {final_err_y:.2f} px")
    print("="*40 + "\n")
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
    integ_max = 1500.0 # Anti-Windup Limit
    
    # Takeoff Sequence
    print("-- Arming & Takeoff")
    # async for health in drone.telemetry.health():
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_local_position_ok and health.is_home_position_ok:
            print("-- Check unlocked, global/local position and home position: OK")
        break
    print("-- Arming & Takeoff")
    try:
        await drone.action.arm()
    except Exception as e:
        print(f"!!! Critical error: {e} !!!")
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
        'pos_x_est': [],  # Estimated position from Kalman
        'pos_y_est': [],
        'vel_x_cmd': [],  # Sent command
        'vel_y_cmd': [],
        'target_visible': [], # 1 if seen, 0 if lost
        'time': [],
        'setpoint_x': [],   # Centro dell'immagine (obiettivo)
        'raw_x': [],        # Lettura nuda e cruda di OpenCV
        'filtered_x': [],   # Lettura dopo il tuo filtro
        'cmd_y': []
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
        CAMERA_OFFSET_Y = 0.04   # Camera offset forward from Center of Mass 
        CAMERA_OFFSET_Z = 0.16   # Camera offset downward from Center of Mass 
        FOCAL_LENGTH    = 550.0 # Focal length in pixels (for 720p/1080p; for 640x480 use ~550)

        # 1. Calculate Effective Camera Altitude
        # If the camera is below the COM, the effective altitude for parallax is lower.
        cam_alt = max(current_alt - CAMERA_OFFSET_Z, 1) 
        
        # 2. Offset pixel to meter conversion (parallax)
        expected_pixel_offset = (CAMERA_OFFSET_Y * FOCAL_LENGTH) / cam_alt
        
        # 3. Application of correction
        # If the camera is forward of the COM, the target appears shifted in the opposite direction of the movement, so we subtract the expected pixel offset from the estimated position to get a more accurate error for control.
        est_x = est_x 
        est_y = est_y + expected_pixel_offset # No correction needed on Y for forward offset
        
        # --- Control Section ---
        cmd_x, cmd_y, cmd_z = 0.0, 0.0, 0.0
        
        # State 1: Takeoff
        if not cruise_altitude_reached:
            if current_alt >= TARGET_ALTITUDE - 0.5:
                print("--- ALTITUDE REACHED ---")
                cruise_altitude_reached = True
                last_seen_time = time.time() # Reset sight timer
            else:
                cmd_z = -1.0
                if measurement is not None: # Preventive centering
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
                err_dist = np.hypot(est_x, est_y)
                dampener = np.clip((current_alt - 0.5) / 1.2, 0.4, 1.0)
                max_speed_xy = np.clip(current_alt * 0.8, 0.1, 1.4)
                # --- COMPLETE PID CALCULATION (P + I + D + FF) ---
                # We calculate the funnel threshold FIRST to use it in all subsequent logic.
                cone_multiplier = np.interp(current_alt, [0.7, 2.0, TARGET_ALTITUDE], [1, 1.25, 1.5])
                current_align_thresh = ALIGN_THRESHOLD * cone_multiplier
                is_aligned = (abs(est_x) < current_align_thresh and abs(est_y) < current_align_thresh)
                MAX_FF = 0.0035
                # --- 2. INTEGRAL ZONE (Dynamic) ---
                # The integral adapts to the alignment threshold with an extra 20% margin.
                # Prevents stalling mid-air by allowing the integral to build up during realignments!
                # --- Freeze Integral in Last Meter ---
                i_dampener = np.clip((current_alt - 1) / 2, 0.3, 1.0)
                spatial_multiplier = np.clip((err_dist - 50.0) / 85.0, 1.0, 0.0)
                # The error is multiplied by the dampener before being added.
                # This way, near the ground it stops accumulating new errors, 
                # but preserves the total value reached perfectly.
                if current_alt < 0.8:
                    i_dampener = 0.0
                spatial_ff_gain = MAX_FF * spatial_multiplier
                integ_x += (est_x * DT) * i_dampener
                integ_y += (est_y * DT) * i_dampener
                    # Standard Anti-Windup
                integ_x = np.clip(integ_x, -integ_max, integ_max)
                integ_y = np.clip(integ_y, -integ_max, integ_max)
                
                # 3. Feed-Forward Gain (Velocity Estimate)
                

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
                          (est_vy * (spatial_ff_gain)))
                
                # --- End PID Calculation ---
# In the landing zone, we cannot assure all pixels as before, so we set a threshold
                
                # Clamping
                cmd_x = np.clip(cmd_x, -max_speed_xy, max_speed_xy)
                cmd_y = np.clip(cmd_y, -max_speed_xy, max_speed_xy)

                # Descent Management
                
                # --- Logging (Inside the Loop) ---
                current_log_time = time.time() - start_log_time
                
                # 1. Ricostruzione dei dati per il grafico (riportandoli in coordinate Pixel 0-640)
                setpoint_pixel = CAM_W / 2
                
                # measurement[0][0] contiene l'errore 'cx' calcolato in vision_callback.
                # Aggiungiamo il centro immagine per avere la coordinata pixel assoluta grezza.
                raw_pixel_x = (measurement[0][0] + setpoint_pixel) if measurement is not None else np.nan
                
                # est_x contiene l'errore filtrato dal tuo LandingKalmanFilter.
                # Aggiungiamo il centro immagine per avere la coordinata pixel assoluta filtrata.
                filtered_pixel_x = est_x + setpoint_pixel

                # 2. Salvataggio
                log_data['time'].append(current_log_time)
                log_data['alt'].append(current_alt)
                log_data['pos_x_est'].append(est_x) 
                log_data['pos_y_est'].append(est_y)
                log_data['vel_x_cmd'].append(cmd_x)
                log_data['vel_y_cmd'].append(cmd_y)
                log_data['target_visible'].append(1 if target_visible else 0)
                
                # 3. Dati specifici per il grafico di Tuning
                log_data['setpoint_x'].append(setpoint_pixel)
                log_data['raw_x'].append(raw_pixel_x)
                log_data['filtered_x'].append(filtered_pixel_x)
                log_data['cmd_y'].append(cmd_y)
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
                    # If we lost target, we might be too low to see it again. To avoid getting stuck in a blind spot, we will command a slow ascent until we reach a certain ceiling where we can search effectively.
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
             #await drone.action.kill()
             await drone.action.land()
             break

        # --- Send Commands ---
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
        print("\n!!! Interrupted by user !!!")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Now we check if log_data actually has data before plotting
        if 'time' in log_data and len(log_data['time']) > 0:    
            import pandas as pd
            import time
            
            # Create a unique filename using system time
            timestamp = int(time.time())
            filename = f"log_volo_{timestamp}.csv"
            
            # Save data in CSV format
            df = pd.DataFrame(log_data)
            df.to_csv(filename, index=False)
            
            print("\n" + "="*40)
            print(f" FLIGHT COMPLETED - DATA SAVED IN: ")
            print(f" {filename}")
            print("="*40 + "\n")
            plot_results(log_data)
        else:
            print("No data recorded to plot.")
            
        print("Cleaning and closing...")
        # This forces the closure of hanging threads of MAVSDK/OpenCV
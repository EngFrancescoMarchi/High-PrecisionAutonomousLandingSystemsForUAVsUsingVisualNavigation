import sys
import cv2
import numpy as np
import time
import asyncio
from mavsdk import System
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.pose_pb2 import Pose
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed, PositionNedYaw)

try:
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
except ImportError:
    print("⚠️ ERRORE CRITICO: Librerie Gazebo non trovate. Fai 'source' prima di lanciare!")
    sys.exit(1)

FREQ = 30.0             
DT = 1.0 / FREQ        
TARGET_ALTITUDE = 10.0   # Quota di crociera (metri)
ALIGN_THRESHOLD = 80    # Pixel tolleranza per iniziare discesa

# Camera Params (gz_x500_vision standard + HD)
CAM_W, CAM_H = 1280, 720
CENTER_X, CENTER_Y = CAM_W // 2, CAM_H // 2

# --- CLASSE ZOH (BUFFER CONDIVISO) ---
class SharedBuffer:
    def __init__(self):
        self.measurement = None 
        self.new_data = False
        self.last_receive_time = 0.0

    def write(self, u, v):
        self.measurement = np.array([[u], [v]])
        self.new_data = True
        self.last_receive_time = time.time()

    def read(self):
        data = self.measurement
        is_fresh = self.new_data
        self.new_data = False
        return data, is_fresh

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
        
        # Opzionale: Rimuovi conversioni colore se vuoi massimizzare FPS
        frame_display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        
        if ids is not None:
            ids_list = ids.flatten().tolist()
            if 0 in ids_list:
                idx = ids_list.index(0)
                target_id = 0
            elif 4 in ids_list:
                idx = ids_list.index(4)
                target_id = 4
            else:
                idx = -1

            if idx != -1:
                c = corners[idx][0]
                cx = int(np.mean(c[:, 0])) - CENTER_X
                cy = int(np.mean(c[:, 1])) - CENTER_Y
                shared_buffer.write(cx, cy) 
                
        # DISABILITATO PER PREVENIRE CORE DUMP IN GAZEBO TRANSPORT
        cv2.imshow("Drone View", frame_display)
        cv2.waitKey(1)
    except Exception:
        pass

# --- TELEMETRIA BACKGROUND ---
current_alt = 0.0
async def telemetry_loop(drone):
    global current_alt
    async for pos in drone.telemetry.position():
        current_alt = pos.relative_altitude_m

# --- MAIN LOOP ---
async def run():
    global current_alt
    
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

    # PID Gains (40Hz + HD)
    KP_X, KD_X = 0.0005, 0.002 
    KP_Y, KD_Y = 0.0005, 0.002
    KI = 0.00035   # <--- NUOVO: Integrale Gain
    
    cruise_altitude_reached = False 
    
    # --- VARIABILI RICOGNIZIONE (SEARCH MODE) ---
    search_active = False
    last_seen_time = time.time()
    search_start_time = 0
    search_leg_index = 0
    search_leg_duration = 1.5 
    base_search_speed = 1.0  
    
    # --- VARIABILI INTEGRALI ---
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

    while True:
    
        measurement, is_new = shared_buffer.read()
        
        # Kalman Prediction & Update
        est_state = kf.predict() 
        if is_new and measurement is not None:
            kf.update(measurement)
        
        est_x, est_vx = est_state[0][0], est_state[1][0]
        est_y, est_vy = est_state[2][0], est_state[3][0]
        
        # --- CORREZIONE PARALLASSE (GEOMETRICA) ---
        # Configurazione fisica del tuo drone:
        CAMERA_OFFSET_X = 0.10   # Metri (Camera spostata in AVANTI rispetto al centro)
        CAMERA_OFFSET_Z = 0.05   # Metri (Camera più BASSA del centro del drone)
        FOCAL_LENGTH    = 1100.0 # Pixel (Standard per HD 720p/1080p in Gazebo. Se 640x480 usa ~550)

        # 1. Calcoliamo la quota reale della camera (non del drone)
        # Se la camera è appesa sotto, è più vicina a terra!
        # Questo è CRUCIALE negli ultimi 50cm.
        cam_alt = max(current_alt - CAMERA_OFFSET_Z, 0.1) 
        
        # 2. Calcolo Offset Pixel atteso
        # Formula: Pixel = (Metri * Focale) / Quota
        expected_pixel_offset = (CAMERA_OFFSET_X * FOCAL_LENGTH) / cam_alt
        
        # 3. Applicazione
        # Se la camera è avanti (X+), vede il target "indietro" (Y+ nell'immagine).
        # Dobbiamo togliere questo offset "naturale" per avere 0 errore al centro.
        est_y = est_y - expected_pixel_offset
        # --- B. MACCHINA A STATI ---
        cmd_x, cmd_y, cmd_z = 0.0, 0.0, 0.0
        
        # STATO 1: SALITA
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

        # STATO 2: DISCESA & RICERCA
        else:
            # Check se abbiamo il target ORA
            target_visible = (measurement is not None)
            
            # --- 2A. TARGET VISIBILE: ATTACCO ---
            if target_visible:
                # Reset Ricerca
                if search_active:
                    print(">>> TARGET AGGANCIATO! STOP RICERCA <<<")
                    search_active = False
                    search_leg_index = 0
                    # Reset Integrale al ritrovamento per evitare scatti
                    integ_x, integ_y = 0.0, 0.0
                last_seen_time = time.time()
            #Damper is the scale of the calculated force, 
            # in this case we will use 40% of calculated, avoid shaking
                # Gain Scheduling
                if current_alt < 2.0:
                    dampener = 0.15
                    max_speed_xy = 0.4 
                else:
                    dampener = 1.0
                    max_speed_xy = 1.1

                # --- CALCOLO PID COMPLETO (P + I + D + FF) ---
                
                # --- GESTIONE INTEGRALE (FREEZE LOGIC) ---
                INTEGRAL_CUTOFF_HEIGHT = 2.0
                
                if current_alt > INTEGRAL_CUTOFF_HEIGHT:
                    # FASE DI VOLO: Accumula e impara il vento
                    integ_x += est_x * DT
                    integ_y += est_y * DT
                    
                    # Anti-Windup standard
                    integ_x = np.clip(integ_x, -integ_max, integ_max)
                    integ_y = np.clip(integ_y, -integ_max, integ_max)
                else:
                    # FASE DI ATTERRAGGIO (< 3m): FREEZE!
                    # Manteniamo il valore attuale per contrastare il vento laterale
                    # senza reagire alle oscillazioni dell'ultimo secondo.
                    pass
                # 3. Feed-Forward Gain (Stima Velocità)
                ff_gain = 0.003 

                # 4. Somma Totale
                # Asse Y (Roll) controlla errore X pixel
                cmd_y = (est_x * KP_X * dampener) + \
                        (est_vx * KD_X * dampener) + \
                        (integ_x * KI) + \
                        (est_vx * ff_gain)
                
                # Asse X (Pitch) controlla errore Y pixel (invertito)
                cmd_x = -((est_y * KP_Y * dampener) + \
                          (est_vy * KD_Y * dampener) + \
                          (integ_y * KI) + \
                          (est_vy * (ff_gain)))
                
                # --- FINE CALCOLO PID ---
#In the landing zone we cannot assure all the pixel as before, so we will set a treshhold
                
                # Clamping
                cmd_x = np.clip(cmd_x, -max_speed_xy, max_speed_xy)
                cmd_y = np.clip(cmd_y, -max_speed_xy, max_speed_xy)

                # Gestione Discesa
                current_align_thresh = ALIGN_THRESHOLD if current_alt > 2.5 else (ALIGN_THRESHOLD * 2.5)
                is_aligned = (abs(est_x) < current_align_thresh and abs(est_y) < current_align_thresh)
                
                if is_aligned:
                    # BLIND DROP CHECK (< 1.2m)
                    #if current_alt < 1.50:
                    #   cmd_z = 0.1 # Drop deciso
                      #  # Azzera integrale in fase finale per evitare overshoot al suolo
                      #  integ_x, integ_y = 0.0, 0.0 
                       # cmd_x, cmd_y = 0.0, 0.0 # Ignora PID, vai giù dritto
                    #else:
                    final_descent_speed = 0.15 if current_alt < 2.0 else 0.35
                    cmd_z = final_descent_speed
                else:
                    # Hovering correttivo
                    cmd_z = 0.0 

            # --- 2B. TARGET PERSO: RICOGNIZIONE ---
            else:
                # 1. Reset immediato della memoria Integrale
                # Se non vedi il target, non devi spingere verso una direzione vecchia!
                integ_x, integ_y = 0.0, 0.0
                
                time_since_loss = time.time() - last_seen_time
                
                # Fase 1: Wait (Anti-Glitch) - 1.5 secondi
                # Aspettiamo un attimo prima di impazzire, magari è solo un frame perso
                if time_since_loss < 1.5:
                    cmd_x, cmd_y, cmd_z = 0.0, 0.0, 0.0
                
                # Fase 2: Search Mode (Spirale + Risalita)
                else:
                    if not search_active:
                        print(f">>> PERSO IN ATTERRAGGIO! RISALITA TATTICA <<<")
                        search_active = True
                        search_start_time = time.time()
                        search_leg_index = 0
                        search_leg_duration = 1.0 # Spirale veloce
                    
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
                    # Se perdi il target in basso (< 10m), DEVI salire per allargare il FOV.
                    # Se non metti questo, farai la spirale a 2m da terra e non lo troverai mai.
                    SEARCH_CEILING = 10.0
                    
                    if current_alt < SEARCH_CEILING:
                        cmd_z = -0.8 # Risali deciso (negativo è SU)
                    else:
                        cmd_z = 0.0  # Mantieni quota se sei già alto

        # --- C. TOUCHDOWN ---
        if current_alt < 1.4 and cruise_altitude_reached:
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
    asyncio.run(run())

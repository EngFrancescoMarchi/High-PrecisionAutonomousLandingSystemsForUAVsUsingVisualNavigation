import cv2
import numpy as np
import time
import threading

# Impostazioni Camera
CAM_W, CAM_H = 640, 480
CENTER_X, CENTER_Y = CAM_W // 2, CAM_H // 2
CONTROL_FREQ = 100.0  # Simuliamo la lettura a 50Hz del loop principale
DT = 1.0 / CONTROL_FREQ

# --- BUFFER CONDIVISO (Come nel main) ---
class SharedBuffer:
    def __init__(self):
        self.measurement = None
        self.new_data = False
        self.last_receive_time = 0.0
        self.lock = threading.Lock() # Aggiunto lock per sicurezza col vero multithreading

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

# --- THREAD DELLA VIDEOCAMERA ---
def vision_thread_func():
    global stop_thread
    print("Inizializzazione telecamera fisica USB in modalità HEADLESS...")
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 100)
    if not cap.isOpened():
        print("ERRORE CRITICO: Impossibile aprire la telecamera. Cavo USB collegato?")
        stop_thread = True
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    prev_time = time.time()

    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        cx, cy = None, None

        if ids is not None:
            ids_list = ids.flatten().tolist()
            idx = -1
            
            if 4 in ids_list:
                idx = ids_list.index(4)
            elif 0 in ids_list:
                idx = ids_list.index(0)

            if idx != -1:
                c = corners[idx][0]
                
                # --- CALCOLO VERO CENTRO PROIETTIVO (Dal tuo codice) ---
                p0, p1, p2, p3 = c[0], c[1], c[2], c[3]
                
                A1 = p2[1] - p0[1]
                B1 = p0[0] - p2[0]
                C1 = A1 * p0[0] + B1 * p0[1]
                
                A2 = p3[1] - p1[1]
                B2 = p1[0] - p3[0]
                C2 = A2 * p1[0] + B2 * p1[1]
                
                det = A1 * B2 - A2 * B1
                
                if det != 0:
                    true_cx = (B2 * C1 - B1 * C2) / det
                    true_cy = (A1 * C2 - A2 * C1) / det
                else:
                    true_cx, true_cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                
                cx = int(true_cx) - CENTER_X
                cy = int(true_cy) - CENTER_Y

        # Scriviamo nel buffer indipendente dal fatto che ci sia o meno il target
        shared_buffer.write(cx, cy)

    cap.release()
    print("Thread visione terminato.")

# --- MAIN LOOP (Simula il loop di controllo) ---
def main():
    global stop_thread
    
    # Avvia il thread della fotocamera
    vision_thread = threading.Thread(target=vision_thread_func)
    vision_thread.start()

    print(f"--- TEST VISIONE MULTITHREAD AVVIATO ({CAM_W}x{CAM_H}) ---")
    print(f"Lettura simulata a {CONTROL_FREQ} Hz. Premi Ctrl+C per uscire.")
    
    time.sleep(2) # Attendi che la telecamera si accenda

    if stop_thread:
        return

    next_wake_time = time.time() + DT
    loops = 0
    read_count = 0

    try:
        while True:
            # Leggiamo dal buffer esattamente come fa il Kalman nel codice di volo
            measurement, is_new = shared_buffer.read()
            
            if is_new:
                read_count += 1 # Contiamo quanti frame freschi stiamo ricevendo realmente
            
            # Stampiamo a schermo solo ogni 10 cicli (5 volte al secondo) per non intasare il terminale SSH
            loops += 1
            if loops % 10 == 0:
                if measurement is not None:
                    print(f"[Loop 50Hz] TARGET LOCKED | cx: {int(measurement[0][0]):4d}, cy: {int(measurement[1][0]):4d} | Nuovi Frame visti: {read_count}/10")
                else:
                    print(f"[Loop 50Hz] TARGET LOST   | Nessun dato nel buffer          | Nuovi Frame visti: {read_count}/10")
                read_count = 0 # Resetta il contatore delle letture fresche

            # Sincronizzazione a 50Hz
            sleep_time = next_wake_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_wake_time += DT

    except KeyboardInterrupt:
        print("\nTest interrotto manualmente da France.")
    finally:
        stop_thread = True
        vision_thread.join()
        print("Chiusura pulita completata.")

if __name__ == "__main__":
    main()
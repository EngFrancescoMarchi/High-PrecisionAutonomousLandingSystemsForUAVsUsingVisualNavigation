import cv2
import numpy as np
import time

# Impostazioni Camera
CAM_W, CAM_H = 640, 480
CENTER_X, CENTER_Y = CAM_W // 2, CAM_H // 2

def main():
    print("Inizializzazione telecamera fisica USB in modalità HEADLESS...")
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERRORE CRITICO: Impossibile aprire la telecamera. Cavo USB collegato?")
        return

    # --- NUOVA SINTASSI OPENCV 4.7+ ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    # ----------------------------------

    prev_time = time.time()

    print(f"--- TEST VISIONE AVVIATO ({CAM_W}x{CAM_H}) ---")
    print("Premi Ctrl+C nel terminale per fermare lo script.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # --- NUOVA SINTASSI RILEVAMENTO ---
            corners, ids, _ = detector.detectMarkers(gray)
            # ----------------------------------

            target_locked = False
            target_id = -1
            cx, cy = 0, 0

            if ids is not None:
                ids_list = ids.flatten().tolist()
                idx = -1
                
                # Cerca prima l'ID 0, poi l'ID 4
                if 0 in ids_list:
                    idx = ids_list.index(0)
                    target_id = 0
                elif 4 in ids_list:
                    idx = ids_list.index(4)
                    target_id = 4

                if idx != -1:
                    c = corners[idx][0]
                    cx = int(np.mean(c[:, 0])) - CENTER_X
                    cy = int(np.mean(c[:, 1])) - CENTER_Y
                    target_locked = True

            status = f"LOCKED (ID:{target_id}) | cx:{cx:4d} cy:{cy:4d}" if target_locked else "LOST"
            print(f"FPS: {fps:5.1f} | {status}")

    except KeyboardInterrupt:
        print("\nTest interrotto manualmente da France.")
    finally:
        cap.release()
        print("Telecamera spenta. Chiusura pulita.")

if __name__ == "__main__":
    main()
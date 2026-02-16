import cv2
import numpy as np
import time

# Impostazioni Camera (Le stesse che pretendi di usare in volo)
CAM_W, CAM_H = 1280, 720
CENTER_X, CENTER_Y = CAM_W // 2, CAM_H // 2

def main():
    print("Inizializzazione telecamera fisica USB...")
    # '0' è l'indice standard per la prima webcam USB collegata
    cap = cv2.VideoCapture(0)
    
    # Forza la risoluzione per testare il vero carico sul Jetson
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERRORE CRITICO: Impossibile aprire la telecamera. Hai attaccato il cavo USB, France?")
        return

    # Dizionario ArUco (usiamo esattamente il 4x4_50 del tuo codice originale)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Variabili per calcolo FPS reale
    prev_time = time.time()

    print(f"--- TEST VISIONE AVVIATO ({CAM_W}x{CAM_H}) ---")
    print("Premi 'q' sulla finestra video per uscire o Ctrl+C nel terminale.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame perso dal buffer hardware! Controlla la connessione USB.")
            time.sleep(0.1)
            continue

        # Calcolo FPS effettivo (vitale per capire se il Jetson sta soffocando)
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        target_locked = False
        target_id = -1
        cx, cy = 0, 0

        if ids is not None:
            ids_list = ids.flatten().tolist()
            idx = -1
            if 4 in ids_list:
                idx = ids_list.index(4)
                target_id = 4
            elif 0 in ids_list:
                idx = ids_list.index(0)
                target_id = 0

            if idx != -1:
                c = corners[idx][0]
                cx = int(np.mean(c[:, 0])) - CENTER_X
                cy = int(np.mean(c[:, 1])) - CENTER_Y
                target_locked = True
                
                # Disegna il centro e i bordi per darti un feedback visivo
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.circle(frame, (cx + CENTER_X, cy + CENTER_Y), 5, (0, 255, 0), -1)

        # Stampa i risultati a terminale
        status = f"LOCKED (ID:{target_id}) | cx:{cx:4d} cy:{cy:4d}" if target_locked else "LOST"
        print(f"FPS Reali: {fps:5.1f} | {status}")

        # Mostra a schermo il video
        cv2.putText(frame, f"FPS: {fps:.1f} | {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        try:
            cv2.imshow("Jetson Vision Test", frame)
        except Exception as e:
            # Cattura l'errore se provi ad aprire una finestra grafica via SSH senza X11
            print(f"Avviso: Impossibile mostrare la finestra video (sei in SSH senza GUI?). Errore: {e}")
            pass

        # Uscita pulita
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Pulizia rigorosa
    cap.release()
    cv2.destroyAllWindows()
    print("Test concluso.")

if __name__ == "__main__":
    main()
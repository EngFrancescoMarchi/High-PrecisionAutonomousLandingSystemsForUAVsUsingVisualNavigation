import sys
import cv2
import numpy as np
import time

# Import librerie Gazebo
try:
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
except ImportError:
    print("ERRORE: Librerie Gazebo non trovate nel path.")
    sys.exit(1)

TOPIC_NAME = "/camera"

# --- CONFIGURAZIONE ARUCO ---
# Usiamo il dizionario standard
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
# Rimuoviamo i parametri custom per evitare crash sulla versione apt
# aruco_params = cv2.aruco.DetectorParameters() 

def cb(msg):
    try:
        # 1. Decodifica Immagine
        width = msg.width
        height = msg.height
        
        # Copia profonda dei dati per evitare problemi di puntatori
        img_buf = np.frombuffer(msg.data, dtype=np.uint8)
        
        # Reshape
        image = img_buf.reshape((height, width, 3))
        
        # 2. SANITIZZAZIONE MEMORIA (Il Fix per il SIGSEGV)
        # Convertiamo in BGR e forziamo la contiguità in memoria
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame = np.ascontiguousarray(frame)
        
        # Scala di grigi sicura
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray) # DOPPIA SICUREZZA
        
        # 3. Detection (Senza parametri opzionali per ora)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)

        # 4. Disegna i risultati
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Prendi il primo marker
            c = corners[0][0] 
            center_x = int((c[0][0] + c[2][0]) / 2)
            center_y = int((c[0][1] + c[2][1]) / 2)
            
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            img_center_x, img_center_y = width // 2, height // 2
            cv2.line(frame, (img_center_x, img_center_y), (center_x, center_y), (0, 255, 255), 2)
            
            error_x = center_x - img_center_x
            error_y = center_y - img_center_y
            cv2.putText(frame, f"ID:{ids[0][0]} Err: {error_x}, {error_y}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "SEARCHING MARKERS...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("France Thesis - ArUco Tracking", frame)
        
        # Check per chiusura finestra
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)
        
    except Exception as e:
        print(f"Errore stream: {e}")

def main():
    print(f"--- VISION SYSTEM AVVIATO (SAFE MODE) ---")
    node = Node()
    if not node.subscribe(Image, TOPIC_NAME, cb):
        print(f"Errore subscribe a {TOPIC_NAME}")
        return

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import numpy as np

def create_nested_paper_target():
    IMG_SIZE = 2000
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # --- DIMENSIONI CRITICHE ---
    # Per ingannare l'algoritmo, il "buco" non deve distruggere troppi bit
    BIG_SIZE = 1600         
    SMALL_SIZE = 250        # < 20% di BIG_SIZE (1600 * 0.15 = 240) -> Siamo al limite
    BORDER_RATIO = 1.2      # Bordo bianco attorno al piccolo per staccarlo
    
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

    # 1. MARKER GRANDE (ID 0)
    # Generiamo il marker base
    if hasattr(cv2.aruco, 'generateImageMarker'):
        big = np.zeros((BIG_SIZE, BIG_SIZE), dtype=np.uint8)
        cv2.aruco.generateImageMarker(aruco_dict, 0, BIG_SIZE, big, 1)
    else:
        big = cv2.aruco.drawMarker(aruco_dict, 0, BIG_SIZE, 1)
    
    # Incolliamo
    start = (IMG_SIZE - BIG_SIZE) // 2
    img[start:start+BIG_SIZE, start:start+BIG_SIZE] = big

    # 2. AREA DI SICUREZZA (Bordo Bianco)
    # Creiamo un quadrato bianco al centro per pulire i bit "sporchi"
    hole_size = int(SMALL_SIZE * BORDER_RATIO)
    h_start = cy - (hole_size // 2)
    img[h_start:h_start+hole_size, h_start:h_start+hole_size] = 255

    # 3. MARKER PICCOLO (ID 4)
    if hasattr(cv2.aruco, 'generateImageMarker'):
        small = np.zeros((SMALL_SIZE, SMALL_SIZE), dtype=np.uint8)
        cv2.aruco.generateImageMarker(aruco_dict, 4, SMALL_SIZE, small, 1)
    else:
        small = cv2.aruco.drawMarker(aruco_dict, 4, SMALL_SIZE, 1)
        
    s_start = cy - (SMALL_SIZE // 2)
    img[s_start:s_start+SMALL_SIZE, s_start:s_start+SMALL_SIZE] = small

    # Salva
    filename = "nested_paper_target.png"
    cv2.imwrite(filename, img)
    print(f"Creato {filename} - Proporzione: {SMALL_SIZE/BIG_SIZE:.2f} (OK se < 0.2)")
    
    # Preview
    preview = cv2.resize(img, (600, 600))
    cv2.imshow("Nested Target (Paper Style)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_nested_paper_target()
import sys
import cv2
import numpy as np
import time

try:
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
except ImportError:
    print("ERROR: Gazebo libraries not found in path.")
    sys.exit(1)

TOPIC_NAME = "/camera"

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)

def cb(msg):
    try:
        # 1. Image Conversion
        width = msg.width
        height = msg.height
        
        # Copy of data
        img_buf = np.frombuffer(msg.data, dtype=np.uint8)
        
        # Reshape
        image = img_buf.reshape((height, width, 3))
        
        # 2. Conversion from RGB to BGR (OpenCV standard)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame = np.ascontiguousarray(frame)
        
        # Safe grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray) # DOUBLE SAFETY
        
        # 3. Detection of ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)

        # 4. Drawing and error calculation
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # First marker
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
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)
        
    except Exception as e:
        print(f"Stream error: {e}")

def main():
    print("--- VISION SYSTEM STARTED (SAFE MODE) ---")
    node = Node()
    if not node.subscribe(Image, TOPIC_NAME, cb):
        print(f"Error subscribing to {TOPIC_NAME}")
        return

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
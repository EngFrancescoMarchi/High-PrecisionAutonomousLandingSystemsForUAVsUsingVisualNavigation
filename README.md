# Thesis
# High-Precision Autonomous Landing System for UAVs

Questo progetto implementa un sistema di atterraggio di precisione autonomo per droni, basato su visione artificiale (marker ArUco) e protocollo MAVLink. Il sistema è progettato per operare su una *Companion Computer* (NVIDIA Jetson Nano) interfacciata con un controllore di volo (Pixhawk) tramite telemetria seriale e MAVSDK.

## 🏗 Architettura di Sistema

Il sistema si divide in due ambienti operativi: la simulazione (Gazebo) e l'hardware fisico. 

### Componenti Hardware
* **Flight Controller:** Pixhawk (PX4 Firmware)
* **Companion Computer:** NVIDIA Jetson Nano
* **Sensore Visivo:** Telecamera USB standard (calibrata a 640x480 @ 30fps per ottimizzare il carico CPU)
* **Ground Control Station (GCS):** PC remoto con QGroundControl su OS Linux.

### Rete e Routing (MAVLink)
Le comunicazioni tra il Flight Controller, la Companion Computer e la GCS sono gestite da `mavlink-router`, configurato per instradare i pacchetti dalla porta seriale (`/dev/ttyTHS1`) verso due endpoint UDP:
* `127.0.0.1:14540` -> Porta locale per i comandi di volo inviati via MAVSDK in Python.
* `IP_GCS:14550` -> Porta esterna per la telemetria verso QGroundControl.

---

## 📂 Struttura del Codice

Il repository contiene tre file principali, ognuno con uno scopo specifico per lo sviluppo e il collaudo del drone:

### 1. `test_visione.py` (Hardware Validation)
Script di collaudo per l'hardware fisico. Legge il flusso video direttamente dalla telecamera USB collegata al Jetson (`cv2.VideoCapture(0)`) e utilizza OpenCV (v4.7+) per rilevare i marker ArUco (Dizionario `DICT_4X4_50`). 
* **Headless Mode:** Progettato per essere eseguito via SSH, non renderizza finestre grafiche (`cv2.imshow` disabilitato) per evitare crash del server X11 e risparmiare risorse.
* Restituisce a terminale gli FPS in tempo reale e le coordinate pixel (`cx`, `cy`) dell'errore rispetto al centro del bersaglio.

### 2. `vision_bridge.py` (Simulation Debugging)
Nodo di test progettato per l'ambiente simulato. 
* Si iscrive al topic `/camera` di Gazebo tramite `gz.transport13`.
* Converte il flusso di byte delle immagini in array NumPy compatibili con OpenCV (da RGB a BGR e scala di grigi).
* Calcola l'errore visivo e disegna i vettori e i centri a schermo per un feedback visivo immediato durante le simulazioni.

### 3. `landing_controller.py` (Mission Core)
Il cervello del sistema di atterraggio. Questo script unisce la computer vision al controllo di volo tramite **MAVSDK**.
* **Filtro di Kalman:** Stima e pulisce le misurazioni di posizione in coordinate pixel, mitigando perdite di frame o disturbi visivi. Incorpora una logica "Zero-Order Hold" (ZOH) per gestire i frame persi.
* **Correzione Parallasse:** Calcola l'offset della telecamera rispetto al centro di massa del drone per evitare atterraggi disallineati.
* **Controller PID:** Genera le velocità `cmd_x` e `cmd_y` in base all'errore visivo, con guadagni dinamici (Gain Scheduling) che diventano più conservativi man mano che l'altitudine diminuisce. Include una logica Anti-Windup per l'azione integrale.
* **Search Mode:** Se il target viene perso visivamente per più di 1.5 secondi, il drone avvia una manovra a spirale e risale a quota sicura (`SEARCH_CEILING`) per tentare di riacquisire visivamente il marker.

---

## 🚀 Guida all'Avvio (Test Fisico)

1. **Setup Rete:** Assicurarsi che GCS e Jetson siano sulla stessa rete LAN/Hotspot.
2. **Routing MAVLink:** Sul Jetson, avviare il router per aprire le porte:
   ```bash
   mavlink-routerd -e IP_GCS:14550 -e 127.0.0.1:14540 /dev/ttyTHS1:1000000
Finito simulazione:
1. Architettura di Controllo e Stima (Il "Cervello")Siamo passati da un loop di controllo pigro a 25 Hz a una configurazione multi-rate a 100 Hz.Frequenza di Controllo: Il loop principale ora gira a $100\text{ Hz}$, inviando setpoint fluidi al Pixhawk per azzerare la latenza di risposta dei motori.Filtro di Kalman Multi-Rate: Il filtro esegue la fase di Predict a $100\text{ Hz}$ per mantenere la stima fluida, mentre la fase di Update avviene solo quando la telecamera fornisce un nuovo frame (~30 Hz).Stabilità Finale: Abbiamo rimosso le oscillazioni distruttive a bassa quota sostituendo il tuo vecchio "scalino" di potenza con un Gain Scheduling lineare che riduce gradualmente l'aggressività dei PID man mano che il drone si avvicina a terra.Deadband: Introdotta una zona morta di 20 pixel sotto i 60 cm per evitare che il drone insegua il rumore visivo negli ultimi istanti di volo.2. Computer Vision e Ottimizzazione "Anti-Blur" (Gli "Occhi")Abbiamo smesso di trattare la webcam Nexigo come una fotocamera per selfie e l'abbiamo resa un sensore di navigazione serio.Centro Proiettivo: Il calcolo dell'errore non usa più una media banale, ma l'intersezione delle diagonali del marker ArUco per una precisione millimetrica anche con angolazioni spinte.Shutter Manuale: Per eliminare il motion blur (la sfocatura che faceva perdere il target nei movimenti veloci), abbiamo disattivato l'auto-esposizione e bloccato il tempo di scatto a valori rapidi via OpenCV/v4l2.Validazione FPS: I test hardware hanno confermato che, nonostante il carico della Jetson Nano, la visione mantiene $30\text{ FPS}$ costanti all'interno di un loop di controllo a $100\text{ Hz}$.3. Connettività e Configurazione Hardware (Il "Ponte")Dopo i vari tentativi al buio, abbiamo finalmente stabilito la comunicazione tra le schede.Il Link Seriale: Abbiamo identificato correttamente la porta TELEM 2 del Pixhawk 6C come porta di comunicazione onboard.Allineamento Baud Rate: Abbiamo risolto l'incongruenza della velocità di comunicazione, sincronizzando Jetson e Pixhawk a 1.000.000 baud.Parametri PX4: In QGroundControl abbiamo impostato MAV_1_CONFIG su Telem 2 e MAV_1_MODE su Onboard, ottimizzando il traffico MAVLink per la companion computer.Permessi Linux: Abbiamo liberato la porta /dev/ttyTHS1 dai processi di sistema (come nvgetty) per permettere al tuo utente di parlare direttamente col drone.
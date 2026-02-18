import asyncio
from mavsdk import System

async def run():
    drone = System()
    
    print("Tentativo di connessione al Pixhawk su /dev/ttyTHS1 a 1000000 baud...")
    
    # Sostituisci il 921600 con 1000000
    await drone.connect(system_address="serial:///dev/ttyTHS1:1000000")
    print("In attesa della telemetria...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("--- PIXHAWK CONNESSO CON SUCCESSO! ---")
            break

    # Leggiamo l'assetto per confermare che i dati passano
    async for attitude in drone.telemetry.attitude_euler():
        print(f"Roll: {attitude.roll_deg:.2f} | Pitch: {attitude.pitch_deg:.2f} | Yaw: {attitude.yaw_deg:.2f}")
        print("Dati ricevuti! Premi Ctrl+C per uscire.")
        break

if __name__ == "__main__":
    asyncio.run(run())
import asyncio
from mavsdk import System

async def run():
    drone = System()

    print("Attempting to connect to Pixhawk on /dev/ttyTHS1 at 1000000 baud...")

    # Replace 921600 with 1000000
    await drone.connect(system_address="serial:///dev/ttyTHS1:1000000")
    print("Waiting for telemetry...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("--- PIXHAWK CONNECTED SUCCESSFULLY! ---")
            break

    # Reading attitude to confirm data flow
    async for attitude in drone.telemetry.attitude_euler():
        print(f"Roll: {attitude.roll_deg:.2f} | Pitch: {attitude.pitch_deg:.2f} | Yaw: {attitude.yaw_deg:.2f}")
        print("Data received! Press Ctrl+C to exit.")
        break

if __name__ == "__main__":
    asyncio.run(run())
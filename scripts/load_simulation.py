import asyncio
import httpx
import time
import random
from typing import List

# CONFIGURATION
API_URL = "http://localhost:8000/api/v1/predict"
SYMPTOMS_POOL = ["Cough", "Fever", "Dyspnea", "Chest Pain", "Fatigue"]
SAMPLE_IMAGE_PATH = "sample_xray.jpg" # Ensure this exists locally or create a dummy
CONCURRENT_USERS = 5
TOTAL_REQUESTS = 20

async def send_request(client: httpx.AsyncClient, user_id: int):
    """Simulates a single user request."""
    symptoms = ",".join(random.sample(SYMPTOMS_POOL, random.randint(1, 3)))
    
    # Using a dummy byte content if file doesn't exist
    files = {'file': ('test.jpg', b"fake-image-content-at-least-some-bytes" * 100, 'image/jpeg')}
    data = {'symptoms': symptoms}
    
    start = time.time()
    try:
        response = await client.post(API_URL, files=files, data=data, timeout=30.0)
        latency = (time.time() - start) * 1000
        print(f"[User {user_id}] Status: {response.status_code} | Latency: {latency:.2f}ms")
        return latency
    except Exception as e:
        print(f"[User {user_id}] Failed: {e}")
        return None

async def run_simulation():
    """Orchestrates the load simulation."""
    print(f"🚀 Starting simulation: {CONCURRENT_USERS} concurrent users, {TOTAL_REQUESTS} total requests.")
    
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(TOTAL_REQUESTS):
            tasks.append(send_request(client, i))
            if len(tasks) >= CONCURRENT_USERS:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Note: Use this script ONLY when the FastAPI server is running
    # asyncio.run(run_simulation())
    print("Usage: Start your FastAPI server, then uncomment 'asyncio.run(run_simulation())' in this script and run it.")

import asyncio
import time
import httpx

BASE_URL = "http://localhost:8000"


async def send_request(i):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/ticket",
            json={
                "subject": f"Issue {i}",
                "description": "System performance degraded and users cannot login"
            }
        )
        return response.status_code


async def run_load_test(concurrent_requests=20):
    start = time.time()

    tasks = [send_request(i) for i in range(concurrent_requests)]
    results = await asyncio.gather(*tasks)

    duration = time.time() - start

    print(f"\nSent {concurrent_requests} concurrent requests")
    print(f"Total time: {duration:.2f}s")
    print(f"Average per request: {duration/concurrent_requests:.2f}s")
    print(f"Status codes: {results}")


if __name__ == "__main__":
    asyncio.run(run_load_test(20))
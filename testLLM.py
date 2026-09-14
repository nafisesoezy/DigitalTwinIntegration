import os
import requests

base = os.environ.get("LLM_BASE_URL")
key = os.environ.get("LLM_API_KEY")

print("Base URL:", base)
print("Key exists:", bool(key))

r = requests.get(
    f"{base}/api/v0/models",
    headers={"Authorization": f"Bearer {key}"}
)

print("Status:", r.status_code)
print(r.json())
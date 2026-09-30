import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8765"
batch = sys.argv[1]
ids = json.loads(urllib.request.urlopen(f"{BASE}/v1/batches/{batch}").read())["jobs"]
deadline = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 560)
last = {}
while time.time() < deadline:
    st = {i: json.loads(urllib.request.urlopen(f"{BASE}/v1/jobs/{i}").read()) for i in ids}
    cur = {i: v["status"] for i, v in st.items()}
    if cur != last:
        print(json.dumps(cur), flush=True)
        last = cur
    if all(s in ("completed", "failed", "cancelled") for s in cur.values()):
        for i, v in st.items():
            if v["status"] != "completed":
                print(i, v.get("error"), flush=True)
        print("E2E_DONE", flush=True)
        sys.exit(0)
    time.sleep(5)
print("E2E_TIMEOUT", json.dumps(last), flush=True)
sys.exit(1)

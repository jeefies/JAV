import json
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/mnt/data/AV/JAV")
from jav.store import TERMINAL

BASE = "http://127.0.0.1:8765"


def get(url):
    try:
        return json.loads(urllib.request.urlopen(url, timeout=30).read())
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
        print(f"poll error: {e}", flush=True)
        return None


batch = sys.argv[1]
data = get(f"{BASE}/v1/batches/{batch}")
if data is None:
    sys.exit("batch fetch failed")
ids = data["jobs"]
deadline = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 560)
last = {}
while time.time() < deadline:
    st = {i: get(f"{BASE}/v1/jobs/{i}") for i in ids}
    if any(v is None for v in st.values()):
        time.sleep(5)
        continue
    cur = {i: v["status"] for i, v in st.items()}
    if cur != last:
        print(json.dumps(cur), flush=True)
        last = cur
    if all(s in TERMINAL for s in cur.values()):
        for i, v in st.items():
            if v["status"] != "completed":
                print(i, v.get("error"), flush=True)
        print("E2E_DONE", flush=True)
        sys.exit(0)
    time.sleep(5)
print("E2E_TIMEOUT", json.dumps(last), flush=True)
sys.exit(1)

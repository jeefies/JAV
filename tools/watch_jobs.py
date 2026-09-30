import json
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/mnt/data/AV/JAV")
from jav.store import TERMINAL

BASE = "http://127.0.0.1:8765"


def get(url):
    """Poll with socket timeout + error tolerance: one stalled or failing
    response must not wedge or crash the watcher."""
    try:
        return json.loads(urllib.request.urlopen(url, timeout=30).read())
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
        print(f"poll error: {e}", flush=True)
        return None


want = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] != "all" else None
horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 500
jobs = get(BASE + "/v1/jobs?status=queued,running,starting_runtime&limit=50")
if jobs is None:
    sys.exit("initial job list fetch failed")
ids = [j["id"] for j in jobs["jobs"] if want is None or j["workflow"] in want]
print("watching", ids, flush=True)
last, t0 = {}, time.time()
if not ids:
    print("NOTHING_TO_WATCH", flush=True)
    sys.exit(0)
while time.time() - t0 < horizon:
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
        print("WATCH_DONE", flush=True)
        sys.exit(0)
    time.sleep(5)
print("WATCH_TIMEOUT", flush=True)
sys.exit(1)

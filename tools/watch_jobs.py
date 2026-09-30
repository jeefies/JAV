import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8765"
want = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] != "all" else None
horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 500
jobs = json.loads(urllib.request.urlopen(
    BASE + "/v1/jobs?status=queued,running,starting_runtime&limit=50").read())["jobs"]
ids = [j["id"] for j in jobs if want is None or j["workflow"] in want]
print("watching", ids, flush=True)
last, t0 = {}, time.time()
if not ids:
    print("NOTHING_TO_WATCH", flush=True)
    sys.exit(0)
while time.time() - t0 < horizon:
    st = {i: json.loads(urllib.request.urlopen(f"{BASE}/v1/jobs/{i}").read()) for i in ids}
    cur = {i: v["status"] for i, v in st.items()}
    if cur != last:
        print(json.dumps(cur), flush=True)
        last = cur
    if all(s in ("completed", "failed", "cancelled") for s in cur.values()):
        for i, v in st.items():
            if v["status"] != "completed":
                print(i, v.get("error"), flush=True)
        print("WATCH_DONE", flush=True)
        sys.exit(0)
    time.sleep(5)
print("WATCH_TIMEOUT", flush=True)
sys.exit(1)

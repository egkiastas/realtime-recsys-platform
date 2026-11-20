"""Save the Grafana dashboard as docs/img/grafana.png via headless Chrome's DevTools protocol.

Chrome's --screenshot flag fires before Grafana's queries return, so this loads the page,
waits in real time, then captures.

    python scripts/screenshot_dashboard.py [--minutes 15]
"""

import argparse
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=15)
    ap.add_argument("--wait", type=float, default=15)
    args = ap.parse_args()
    url = f"http://localhost:3000/d/recsys/recsys?orgId=1&kiosk&theme=light&from=now-{args.minutes}m&to=now"
    chrome = subprocess.Popen(
        [CHROME, "--headless=new", "--remote-debugging-port=9222", "--window-size=1600,1250", "--hide-scrollbars",
         f"--user-data-dir={tempfile.mkdtemp()}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        for _ in range(50):
            try:
                pages = json.load(urllib.request.urlopen("http://localhost:9222/json"))
                break
            except OSError:
                time.sleep(0.2)
        ws_url = next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page")
        with connect(ws_url, max_size=None) as ws:
            ws.send(json.dumps({"id": 1, "method": "Page.navigate", "params": {"url": url}}))
            time.sleep(args.wait)
            ws.send(json.dumps({"id": 2, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == 2:
                    break
        out = ROOT / "docs" / "img" / "grafana.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(msg["result"]["data"]))
        print(f"saved {out}")
    finally:
        chrome.terminate()


if __name__ == "__main__":
    main()

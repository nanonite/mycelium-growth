#!/usr/bin/env python3
"""Download files from a Zenodo record or Figshare article with resume + md5 check.

Usage:
  scripts/fetch.py zenodo 15224240 myceliumseg [--only a.zip b.zip] [--list]
  scripts/fetch.py figshare 22666138 podospora_thallus

Writes data/raw/<name>/<file>, data/raw/<name>/SOURCE.md, and appends to data/manifest.csv.
"""
import argparse, csv, datetime, hashlib, json, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"
CHUNK = 8 << 20
STALL_WINDOW = 180              # s; reconnect + resume if throughput drops below
STALL_MIN_BYTES = 9_000_000     # ~50 kB/s over the window


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def zenodo(rec):
    d = get_json(f"https://zenodo.org/api/records/{rec}")
    files = [dict(name=f["key"], size=f["size"], md5=f["checksum"].removeprefix("md5:"),
                  url=f["links"]["self"]) for f in d["files"]]
    meta = dict(title=d["metadata"]["title"], doi=d.get("doi", ""),
                license=d["metadata"].get("license", {}).get("id", "unknown"),
                page=f"https://zenodo.org/records/{rec}")
    return meta, files


def figshare(art):
    d = get_json(f"https://api.figshare.com/v2/articles/{art}")
    files = [dict(name=f["name"], size=f["size"], md5=f.get("computed_md5") or "",
                  url=f["download_url"]) for f in d["files"]]
    meta = dict(title=d["title"], doi=d.get("doi", ""),
                license=(d.get("license") or {}).get("name", "unknown"), page=d.get("url_public_html", ""))
    return meta, files


def md5sum(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def download(f, dest, retries=20):
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= f["size"]:
            break
        req = urllib.request.Request(f["url"], headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:      # server ignored Range → restart
                    have = 0
                    part.unlink()
                t0, last = time.time(), 0.0
                win_t, win_b = time.time(), have   # stall watchdog window
                with open(part, "ab") as out:
                    # read1 returns whatever has arrived; read(n) would block until n bytes
                    for b in iter(lambda: r.read1(CHUNK), b""):
                        out.write(b)
                        have += len(b)
                        if time.time() - win_t > STALL_WINDOW:
                            if have - win_b < STALL_MIN_BYTES:
                                raise TimeoutError(f"stalled: {(have - win_b)/1e3:.0f} kB in {STALL_WINDOW}s")
                            win_t, win_b = time.time(), have
                        if time.time() - last > 60:   # one status line per minute
                            last = time.time()
                            rate = (have / (time.time() - t0 + 1e-9)) / 1e6
                            print(f"  {f['name']}: {have/1e9:.2f}/{f['size']/1e9:.2f} GB ({rate:.1f} MB/s)", flush=True)
        except Exception as e:
            print(f"  attempt {attempt} failed: {e}; retrying", flush=True)
            time.sleep(10 * attempt)
    if not part.exists() or part.stat().st_size < f["size"]:
        raise SystemExit(f"INCOMPLETE after {retries} attempts: {f['name']} (rerun to resume)")
    if f["md5"] and md5sum(part) != f["md5"]:
        part.rename(dest.with_suffix(dest.suffix + ".bad"))
        raise SystemExit(f"MD5 MISMATCH: {f['name']}")
    part.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["zenodo", "figshare"])
    ap.add_argument("id")
    ap.add_argument("name")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    meta, files = (zenodo if a.source == "zenodo" else figshare)(a.id)
    if a.only:
        missing = set(a.only) - {f["name"] for f in files}
        if missing:
            raise SystemExit(f"not in record: {missing}")
        files = [f for f in files if f["name"] in a.only]
    if a.list:
        print(f"{meta['title']} | {meta['license']}")
        for f in files:
            print(f"  {f['name']}  {f['size']/1e9:.2f} GB  md5={f['md5']}")
        return

    out = RAW / a.name
    out.mkdir(parents=True, exist_ok=True)
    new_manifest = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as mf:
        w = csv.writer(mf)
        if new_manifest:
            w.writerow(["dataset", "source", "id", "file", "bytes", "md5", "license", "downloaded"])
        for f in files:
            dest = out / f["name"]
            if dest.exists() and dest.stat().st_size == f["size"]:
                print(f"skip (present): {f['name']}")
                continue
            print(f"get: {f['name']} ({f['size']/1e9:.2f} GB)", flush=True)
            download(f, dest)
            w.writerow([a.name, a.source, a.id, f["name"], f["size"], f["md5"], meta["license"],
                        datetime.date.today().isoformat()])
            mf.flush()
            print(f"ok:  {f['name']} md5 verified", flush=True)

    src = out / "SOURCE.md"
    if not src.exists():
        src.write_text(f"# {meta['title']}\n\n- Source: {a.source} {a.id}\n- Page: {meta['page']}\n"
                       f"- DOI: {meta['doi']}\n- License: {meta['license']}\n"
                       f"- Fetched: {datetime.date.today().isoformat()} via scripts/fetch.py\n")


if __name__ == "__main__":
    sys.exit(main())

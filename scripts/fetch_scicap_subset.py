"""
Fetch a SMALL subset of the SciCap dataset (CrowdAILab/scicap on HuggingFace) for evaluation.

Run this on a machine with HuggingFace access (the cloud build environment cannot reach it):

    pip install huggingface_hub pandas pyarrow pillow
    python scripts/fetch_scicap_subset.py --inspect            # 1. look at the repo layout (no download)
    python scripts/fetch_scicap_subset.py --n 300               # 2. fetch 300 figures into data/scicap_subset/

It never downloads the whole dataset:
  * parquet / jsonl / json data files are read row by row until --n records are collected
    (json files are downloaded, so the smallest test/val annotation file is chosen);
  * images inside .zip archives are read with random access over HTTP (HfFileSystem), so only the
    needed images are transferred, not the archive.

Output (small enough to commit):
    data/scicap_subset/images/<figure_id>.png
    data/scicap_subset/metadata.json   - normalised records + each record's raw fields
    data/scicap_subset/README.md       - source + licence (SciCap is CC BY-NC-SA 4.0)

The dataset schema is discovered at run time; if normalisation misses a field, the "raw" entry of
each record keeps everything so the loader can be adapted.
"""

import argparse
import io
import json
import os
import re
import sys
import zipfile

REPO = "CrowdAILab/scicap"

# Field names we look for in a record (first match wins)
ID_KEYS = ["figure_id", "image_id", "id", "image_file", "image", "file_name", "filename"]
CAPTION_KEYS = ["caption_no_index", "caption", "caption_text", "mention_caption", "text"]
PARAGRAPH_KEYS = ["paragraph", "paragraphs", "mention", "mentions", "context", "referring_paragraphs"]
OCR_KEYS = ["ocr", "ocr_text"]
TYPE_KEYS = ["figure_type", "type", "class"]


def pick(rec, keys):
    for k in keys:
        if k in rec and rec[k] not in (None, "", [], {}):
            return rec[k]
    return None


def as_text(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(as_text(x) for x in v)
    if isinstance(v, dict):
        return " ".join(as_text(x) for x in v.values())
    return str(v)


def list_files():
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(REPO, files_metadata=True)
    return [(s.rfilename, s.size or 0) for s in info.siblings]


def inspect():
    files = list_files()
    total = sum(sz for _, sz in files)
    print(f"{REPO}: {len(files)} files, {total / 1e9:.2f} GB total\n")
    for name, sz in sorted(files):
        print(f"{sz / 1e6:10.1f} MB  {name}")


def split_rank(name):
    n = name.lower()
    if "hidden" in n:  # hidden test split has no captions
        return 99
    for i, s in enumerate(("val", "dev", "test", "train")):
        if s in n:
            return i
    return 9


def split_of(name):
    """'val-metadata.json' -> 'val'; 'val.json' -> 'val'."""
    return re.sub(r"(-metadata)?\.(json|jsonl|parquet)$", "", os.path.basename(name).lower())


def _json_records(path):
    with open(path) as fh:
        obj = json.load(fh)
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        if isinstance(obj.get("annotations"), list):  # COCO style
            imgs = {i.get("id"): i for i in obj.get("images", []) if isinstance(i, dict)}
            return [{**imgs.get(a.get("image_id"), {}), **a} for a in obj["annotations"] if isinstance(a, dict)]
        if isinstance(obj.get("images"), list):
            return [r for r in obj["images"] if isinstance(r, dict)]
        return [{"id": k, **v} if isinstance(v, dict) else {"id": k, "value": v} for k, v in obj.items()]
    return []


def _join_key(rec):
    for k in ("image_id", "id", "figure_id", "file_name", "image_file"):
        if rec.get(k) not in (None, ""):
            return os.path.splitext(os.path.basename(str(rec[k])))[0]
    return None


def iter_merged_json(files, max_records):
    """Merge every json file of the best split (e.g. val.json + val-metadata.json) on the image id."""
    from huggingface_hub import hf_hub_download
    jsons = [(n, s) for n, s in files if n.lower().endswith(".json") and split_rank(n) < 99]
    if not jsons:
        return
    best = min(split_rank(n) for n, _ in jsons)
    split = split_of(min((n for n, _ in jsons if split_rank(n) == best), key=len))
    group = [(n, s) for n, s in jsons if split_of(n) == split]
    merged = {}
    for name, size in sorted(group, key=lambda x: x[1]):
        if size > 2e9:
            print(f"  skipping {name}: too large")
            continue
        print(f"reading {name} ({size / 1e6:.1f} MB)")
        for rec in _json_records(hf_hub_download(REPO, name, repo_type="dataset")):
            k = _join_key(rec)
            if k is None:
                continue
            merged.setdefault(k, {}).update(rec)
    print(f"merged {len(merged)} records from split '{split}' ({', '.join(n for n, _ in group)})")
    for i, (k, rec) in enumerate(merged.items()):
        if i >= max_records:
            return
        yield f"{split} ({', '.join(n for n, _ in group)})", rec


def iter_records(files, max_records):
    """Yield (source_file, record dict) from the smallest test/val data file(s)."""
    from huggingface_hub import HfFileSystem, hf_hub_download
    fs = HfFileSystem()
    if any(n.lower().endswith(".json") and split_rank(n) < 99 for n, _ in files) and \
            not any(n.lower().endswith((".parquet", ".jsonl")) for n, _ in files):
        yield from iter_merged_json(files, max_records)
        return
    data = [(n, s) for n, s in files if re.search(r"\.(parquet|jsonl|json)$", n)
            and "readme" not in n.lower() and split_rank(n) < 99]
    data.sort(key=lambda x: (split_rank(x[0]), x[1]))
    if not data:
        sys.exit("No parquet/json/jsonl files found - run with --inspect and send the listing.")
    count = 0
    for name, size in data:
        print(f"reading records from {name} ({size / 1e6:.1f} MB)")
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq
            with fs.open(f"datasets/{REPO}/{name}", "rb") as fh:
                pf = pq.ParquetFile(fh)
                for batch in pf.iter_batches(batch_size=64):
                    for rec in batch.to_pylist():
                        yield name, rec
                        count += 1
                        if count >= max_records:
                            return
        elif name.endswith(".jsonl"):
            with fs.open(f"datasets/{REPO}/{name}", "r") as fh:
                for line in fh:
                    if line.strip():
                        yield name, json.loads(line)
                        count += 1
                        if count >= max_records:
                            return
        else:
            if size > 2e9:
                print(f"  skipping {name}: {size / 1e9:.1f} GB is too large to download")
                continue
            path = hf_hub_download(REPO, name, repo_type="dataset")
            with open(path) as fh:
                obj = json.load(fh)
            recs = obj
            if isinstance(obj, dict):
                # common layouts: {"annotations": [...], "images": [...]} or {id: record}
                if isinstance(obj.get("annotations"), list):
                    imgs = {i.get("id"): i for i in obj.get("images", []) if isinstance(i, dict)}
                    recs = [{**imgs.get(a.get("image_id"), {}), **a} for a in obj["annotations"]]
                else:
                    recs = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "value": v} for k, v in obj.items()]
            for rec in recs:
                if isinstance(rec, dict):
                    yield name, rec
                    count += 1
                    if count >= max_records:
                        return
        if count >= max_records:
            return


class SpannedZip:
    """
    Minimal reader for split / spanned zip archives (name.z01, name.z02, ..., name.zip), which
    Python's zipfile rejects ("zipfiles that span multiple disks are not supported").

    The parts are treated as one virtual byte stream; the central directory is read from the end,
    and each member is read with ranged reads, so only the needed bytes are transferred.
    ``opener(name)`` must return a seekable binary file object for a part.
    Supports ZIP64 and the stored / deflate compression methods.
    """

    def __init__(self, parts, sizes, opener):
        self.parts, self.sizes, self.opener = parts, sizes, opener
        self.starts = [sum(sizes[:i]) for i in range(len(parts))]
        self._fh = {}
        self.members = {}
        self._read_central_directory()

    def _read(self, pos, n):
        """Read n bytes at a virtual offset, crossing part boundaries if needed."""
        out = b""
        while n > 0:
            i = max(k for k, st in enumerate(self.starts) if st <= pos)
            fh = self._fh.get(i) or self._fh.setdefault(i, self.opener(self.parts[i]))
            off = pos - self.starts[i]
            fh.seek(off)
            chunk = fh.read(min(n, self.sizes[i] - off))
            if not chunk:
                raise IOError("unexpected end of archive part")
            out += chunk
            pos += len(chunk)
            n -= len(chunk)
        return out

    def _virtual(self, disk, offset):
        return self.starts[disk] + offset

    def _read_central_directory(self):
        import struct
        total = sum(self.sizes)
        tail_len = min(total, 1 << 16)
        tail = self._read(total - tail_len, tail_len)
        e = tail.rfind(b"PK\x05\x06")
        if e < 0:
            raise ValueError("end of central directory not found")
        (_, disk_no, cd_disk, _, n_entries, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[e:e + 22])
        loc = tail.rfind(b"PK\x06\x07", 0, e)
        if loc >= 0:  # ZIP64
            _, z64_disk, z64_off, _ = struct.unpack("<IIQI", tail[loc:loc + 20])
            z64 = self._read(self._virtual(z64_disk, z64_off), 56)
            (_, _, _, _, _, cd_disk, _, n_entries, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", z64)
        cd = self._read(self._virtual(cd_disk, cd_off), cd_size)
        p = 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            (method, csize, usize, nlen, xlen, clen, dstart, loff) = (
                struct.unpack("<H", cd[p + 10:p + 12])[0], *struct.unpack("<II", cd[p + 20:p + 28]),
                *struct.unpack("<HHHH", cd[p + 28:p + 36]), struct.unpack("<I", cd[p + 42:p + 46])[0])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            q = 0
            while q + 4 <= len(extra):  # ZIP64 extended information
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    vals, r = [], q + 4
                    for flag in (usize == 0xFFFFFFFF, csize == 0xFFFFFFFF, loff == 0xFFFFFFFF):
                        if flag:
                            vals.append(struct.unpack("<Q", extra[r:r + 8])[0])
                            r += 8
                        else:
                            vals.append(None)
                    usize = vals[0] if vals[0] is not None else usize
                    csize = vals[1] if vals[1] is not None else csize
                    loff = vals[2] if vals[2] is not None else loff
                    if dstart == 0xFFFF:
                        dstart = struct.unpack("<I", extra[r:r + 4])[0]
                q += 4 + hlen
            self.members[name] = (method, csize, dstart, loff)
            p += 46 + nlen + xlen + clen

    def read(self, name):
        import struct
        import zlib
        method, csize, disk, loff = self.members[name]
        pos = self._virtual(disk, loff)
        hdr = self._read(pos, 30)
        nlen, xlen = struct.unpack("<HH", hdr[26:30])
        data = self._read(pos + 30 + nlen + xlen, csize)
        if method == 0:
            return data
        if method == 8:
            return zlib.decompressobj(-15).decompress(data)
        raise ValueError(f"unsupported compression method {method} for {name}")


def find_spanned_parts(base_zip, files):
    """'img-split.zip' -> (['img-split.z01', ..., 'img-split.zip'], sizes) if split parts exist."""
    stem = base_zip[:-4]
    parts = sorted([(n, s) for n, s in files if re.fullmatch(re.escape(stem) + r"\.z\d+", n)],
                   key=lambda x: int(x[0].rsplit(".z", 1)[1]))
    if not parts:
        return None
    sizes = dict(files)
    names = [n for n, _ in parts] + [base_zip]
    return names, [sizes[n] for n in names]


class ZipImageIndex:
    """Random-access lookup of image members inside remote zip archives."""

    def __init__(self, files):
        from huggingface_hub import HfFileSystem
        self.fs = HfFileSystem()
        self.files = files
        self.zips = sorted([n for n, _ in files if n.lower().endswith(".zip")], key=split_rank)
        self._index = None

    def _build(self):
        self._index = {}
        for z in self.zips:
            print(f"indexing {z} (central directory only)")
            spanned = find_spanned_parts(z, self.files)
            try:
                if spanned:
                    names, sizes = spanned
                    print(f"  spanned archive: {len(names)} parts")
                    zf = SpannedZip(names, sizes, lambda n: self.fs.open(f"datasets/{REPO}/{n}", "rb"))
                    members = list(zf.members)
                else:
                    zf = zipfile.ZipFile(self.fs.open(f"datasets/{REPO}/{z}", "rb"))
                    members = zf.namelist()
            except Exception as e:
                print(f"  could not index {z}: {type(e).__name__}: {e}")
                continue
            print(f"  {len(members)} members")
            for member in members:
                base = os.path.basename(member)
                if base:
                    self._index.setdefault(base, (zf, member))
                    self._index.setdefault(os.path.splitext(base)[0], (zf, member))

    def read(self, key):
        if not self.zips:
            return None
        if self._index is None:
            self._build()
        hit = self._index.get(os.path.basename(str(key))) or self._index.get(os.path.splitext(os.path.basename(str(key)))[0])
        if hit is None:
            return None
        zf, member = hit
        return zf.read(member)


def image_bytes(rec, files_set, zips):
    """Image may be embedded (parquet Image feature), a repo path, or a member of a zip archive."""
    from huggingface_hub import hf_hub_download
    img = rec.get("image")
    if isinstance(img, dict) and img.get("bytes"):
        return img["bytes"]
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    ref = pick(rec, ["image_file", "file_name", "filename", "image_path", "image", "figure_id", "id"])
    if ref is None:
        return None
    ref = str(ref)
    for cand in files_set:
        if cand.endswith(ref) or cand.endswith(ref + ".png") or cand.endswith(ref + ".jpg"):
            with open(hf_hub_download(REPO, cand, repo_type="dataset"), "rb") as fh:
                return fh.read()
    return zips.read(ref)


def write_outputs(records, out, final=False):
    meta = {"dataset": REPO, "license": "CC BY-NC-SA 4.0 (SciCap, Hsu et al., 2021)",
            "num_figures": len(records), "complete": final, "records": records}
    tmp = os.path.join(out, "metadata.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(tmp, os.path.join(out, "metadata.json"))
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(f"# SciCap subset\n\n{len(records)} figures from `{REPO}` (HuggingFace), fetched with "
                 "`scripts/fetch_scicap_subset.py`.\n\nSciCap: Hsu, Giles, Huang. *SciCap: Generating Captions "
                 "for Scientific Figures*. Findings of EMNLP 2021. Licensed CC BY-NC-SA 4.0 "
                 "(non-commercial, share-alike, attribution).\n")


def fetch(n, out):
    """
    Saves metadata every 25 figures and on exit (also after an error), skips one bad record instead
    of stopping, and does not re-download images that already exist in ``out/images``.
    """
    from PIL import Image
    files = list_files()
    files_set = [f for f, _ in files]
    zips = ZipImageIndex(files)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    records, misses, errors = [], 0, 0
    try:
        for src, rec in iter_records(files, max_records=n * 3):
            if len(records) >= n:
                break
            try:
                rid = pick(rec, ID_KEYS)
                fid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(rid if rid is not None else f"rec{len(records)}"))
                fid = os.path.splitext(fid)[0]
                img_path = os.path.join(out, "images", f"{fid}.png")
                if not os.path.exists(img_path):
                    raw_img = image_bytes(rec, files_set, zips)
                    if raw_img is None:
                        misses += 1
                        if misses > 20 and not records:
                            print("\nCould not locate images. Diagnostics (please paste all of this):")
                            print("  first record:", json.dumps({k: (str(v)[:120]) for k, v in rec.items()}, indent=1))
                            other = [f"{nm} ({sz / 1e6:.0f} MB)" for nm, sz in files if not nm.lower().endswith(".json")]
                            print("  non-json repo files:", other[:40])
                            sys.exit(1)
                        continue
                    im = Image.open(io.BytesIO(raw_img)).convert("RGB")
                    if max(im.size) > 1024:  # keep the subset small
                        im.thumbnail((1024, 1024))
                    im.save(img_path, optimize=True)
                raw = {k: ("<image>" if k == "image" and isinstance(v, (bytes, bytearray, dict)) else v)
                       for k, v in rec.items()}
                records.append({
                    "figure_id": fid,
                    "image_path": f"images/{fid}.png",
                    "caption": as_text(pick(rec, CAPTION_KEYS)),
                    "paragraph_text": as_text(pick(rec, PARAGRAPH_KEYS)),
                    "ocr_text": as_text(pick(rec, OCR_KEYS)),
                    "figure_type": as_text(pick(rec, TYPE_KEYS)) or "graph_plot",
                    "source_file": src,
                    "raw": json.loads(json.dumps(raw, default=str)),
                })
                print(f"  [{len(records)}/{n}] {fid}")
                if len(records) % 25 == 0:
                    write_outputs(records, out)
            except SystemExit:
                raise
            except Exception as e:
                errors += 1
                print(f"  skipped a record: {type(e).__name__}: {e}")
                if errors > 50:
                    print("  too many errors, stopping")
                    break
    finally:
        if records:
            write_outputs(records, out, final=len(records) >= n)
    size = sum(os.path.getsize(os.path.join(out, "images", f)) for f in os.listdir(os.path.join(out, "images")))
    print(f"\nWrote {len(records)} figures ({size / 1e6:.1f} MB) to {out}  (skipped records: {errors})")
    if records and not records[0]["caption"]:
        print("WARNING: no caption field recognised - the 'raw' entries keep the original fields.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true", help="list repo files and sizes, download nothing")
    ap.add_argument("--n", type=int, default=300, help="number of figures to fetch")
    ap.add_argument("--out", default="data/scicap_subset")
    a = ap.parse_args()
    inspect() if a.inspect else fetch(a.n, a.out)

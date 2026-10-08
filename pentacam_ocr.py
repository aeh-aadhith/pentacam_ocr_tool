#!/usr/bin/env python3
"""
Pentacam printout OCR pipeline.

Runs a locally-hosted DeepSeek-OCR (via llama.cpp) over OCULUS Pentacam
JPG printouts and extracts the labeled numeric fields (K1, K2, Km, KMax,
Pachy, indices, etc.) into CSV/JSON.

Works on three known Pentacam report templates (detected automatically per
image from the page title):
  - "Refractive"                          (e.g. PATIENT_OD.JPG)
  - "Belin/Ambrosio Enhanced Ectasia..."  (e.g. PATIENT_OD1.JPG)
  - "4 Maps Refractive"                   (e.g. PATIENT_OD_4 Maps Refractive.JPG)

It does NOT run OCR on the whole page at once. Full-page OCR was tested
and reliably breaks down (infinite repetition loop) as soon as it reaches
the dense keratometry number cluster next to the circular Rf/Rs polar
diagram -- this happens at every quantization level tested (Q4_K_M, Q8_0)
and with both the grounding and plain-OCR prompts. Cropping the page into
per-box regions that exclude the circular diagrams/color maps avoids the
issue entirely and gives clean, correct output every time.

Usage:
    python3 pentacam_ocr.py <folder_with_jpgs> [--out results]
    python3 pentacam_ocr.py --setup-only        # just set up the runtime

Requirements (Linux):
    pip install pillow
    sudo apt install git cmake build-essential    # or your distro's equivalent

Everything else is set up automatically on first run (one time, ~4 GB
download + a few minutes of compiling) into ~/.local/share/deepseek-ocr
(override with PENTACAM_OCR_HOME):
  - llama-mtmd-cli, compiled from a pinned llama.cpp commit
  - the DeepSeek-OCR Q8_0 GGUF weights + vision projector (SHA-256 checked)
The llama.cpp commit is pinned on purpose: newer llama.cpp releases
process the image differently and were seen to misread table rows that
this commit reads correctly.

Patient images never leave the machine: downloads happen *before* any
image is opened, and the OCR itself runs with networking disabled (see
"Network isolation" below).
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OCR_HOME = Path(os.environ.get("PENTACAM_OCR_HOME") or Path.home() / ".local/share/deepseek-ocr")

# Pinned llama.cpp commit: the final head of ggml-org/llama.cpp#17400, which
# added DeepSeek-OCR support. All crop boxes and parsers were validated
# against this build; later releases changed DeepSeek-OCR image handling and
# produce different (sometimes wrong) output. Don't bump without re-checking.
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_COMMIT = "95cc5665859b49d7158c5c4abc9943adf109c6d5"
LLAMA_BIN_DEFAULT = OCR_HOME / "bin" / "llama-mtmd-cli"

# Pinned model revision on https://huggingface.co/sabafallah/DeepSeek-OCR-GGUF
HF_REVISION = "d26779bcd1cb301fec3ff82adc672f18384776fc"
HF_URL = f"https://huggingface.co/sabafallah/DeepSeek-OCR-GGUF/resolve/{HF_REVISION}/"
MODEL_FILES = {  # name -> sha256
    "deepseek-ocr-q8_0.gguf": "81ede3e256230707dccf7fa052570c3a939d57db99de655f43cbb1a830d14d92",
    "mmproj-deepseek-ocr-bf16.gguf": "4caeed8b6c3c7d25dfebccfdb5cf34d6ae540ef4dc4fa2b9842b69cfa50ecbe2",
}
MODEL = OCR_HOME / "models" / "deepseek-ocr-q8_0.gguf"
MMPROJ = OCR_HOME / "models" / "mmproj-deepseek-ocr-bf16.gguf"

# Set by ensure_runtime(). PENTACAM_OCR_LLAMA_BIN points at a different
# llama-mtmd-cli build instead of the pinned one.
LLAMA_BIN = None

# Seconds allowed per cropped region. Generous because CPU-only OCR on a
# small laptop can take several minutes per region.
OCR_TIMEOUT = int(os.environ.get("PENTACAM_OCR_TIMEOUT", "900"))

# Pixel crop boxes calibrated against 1200x838 Pentacam JPG exports
# (OCULUS software version 1.33r02). If your printouts are a different
# resolution, rescale these boxes proportionally.
TEMPLATE_REGIONS = {
    "refractive": {
        "header": (0, 0, 600, 45),
        "demographics": (0, 45, 400, 195),
        "k_readings": (88, 160, 400, 315),
        "pachy_pupil": (0, 315, 400, 495),
        "indices": (400, 315, 610, 495),
    },
    "belin_ambrosio": {
        "header": (0, 0, 900, 45),
        "demographics": (618, 45, 875, 178),
        "k_readings": (618, 172, 875, 232),  # top at 172: 178 clipped the K1/Axis digits
        "pachy_dist": (618, 232, 875, 283),
        "progression": (618, 283, 875, 350),
        "reference_db": (618, 770, 1200, 838),
    },
    # Left-hand column only; the four color maps are never cropped. The
    # Cornea Front/Back K boxes start at x=100 to cut off the small polar
    # diagram, which is why each is split from its QS/Axis/Q-val rows.
    "four_maps": {
        "header": (0, 0, 900, 45),
        "demographics": (0, 40, 330, 192),
        "front_k": (100, 200, 330, 312),
        "front_indices": (0, 312, 330, 375),
        "back_k": (100, 388, 330, 500),
        "back_indices": (0, 500, 330, 565),
        "pachy": (0, 572, 330, 705),
        "volumes": (0, 705, 330, 830),
    },
}

UPSCALE = 3  # crops are small; upscaling improves OCR reliability


# ---------------------------------------------------------------------------
# Runtime setup (downloads; runs before any patient image is touched)
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, sha256: str):
    """Download url -> dest, resuming a partial .part file across retries
    (large files over flaky connections otherwise restart from zero), then
    verify the SHA-256 before moving it into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 51):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:  # server ignored Range
                    have = 0
                total = have + int(resp.headers.get("Content-Length", 0))
                with open(part, "ab" if have else "wb") as out:
                    done, last = have, 0.0
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 2:
                            pct = f"{100 * done / total:5.1f}%" if total else ""
                            print(f"\r    {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB {pct}",
                                  end="", flush=True)
                            last = time.time()
            print()
            break
        except Exception as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:  # already complete
                break
            print(f"\n    download interrupted ({e}); retrying [{attempt}/50]...", flush=True)
            time.sleep(min(30, 2 * attempt))
    else:
        sys.exit(f"Failed to download {url}")

    print(f"    verifying {dest.name} ...", flush=True)
    if _sha256(part) != sha256:
        part.unlink()
        sys.exit(f"Checksum mismatch for {dest.name} (corrupt download, deleted). Re-run to retry.")
    part.replace(dest)


def _llama_commit_ok(binary: Path) -> bool:
    """True if `binary --version` reports the pinned commit."""
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                             timeout=60, env=dict(os.environ, LD_LIBRARY_PATH=str(binary.parent)))
    except OSError:
        return False
    return LLAMA_COMMIT[:7] in out.stdout + out.stderr


def _build_llama(dest: Path):
    """Fetch the pinned llama.cpp commit and compile llama-mtmd-cli as one
    self-contained binary (static llama/ggml libs, no HTTP client)."""
    compiler = os.environ.get("CXX") or next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    missing = [t for t in ("git", "cmake", "make") if not shutil.which(t)] + ([] if compiler else ["C++ compiler"])
    if missing:
        sys.exit(f"Cannot build the OCR runtime: missing {', '.join(missing)}.\n"
                 f"Install them (Debian/Ubuntu: sudo apt install git cmake build-essential) and re-run.")

    src = OCR_HOME / "llama-src"
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    steps = [
        ["git", "init", "-q"],
        ["git", "fetch", "-q", "--depth", "1", LLAMA_REPO, LLAMA_COMMIT],
        ["git", "checkout", "-q", "FETCH_HEAD"],
        ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
         "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
         "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF"],
        ["cmake", "--build", "build", "-j", str(os.cpu_count() or 2), "--target", "llama-mtmd-cli"],
    ]
    print(f"Building llama.cpp {LLAMA_COMMIT[:9]} (one time, a few minutes) ...", flush=True)
    log = OCR_HOME / "llama-build.log"
    with open(log, "w") as logf:
        for cmd in steps:
            if subprocess.run(cmd, cwd=src, stdout=logf, stderr=subprocess.STDOUT).returncode:
                sys.exit(f"Build step failed: {' '.join(cmd)}\nSee {log}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "build" / "bin" / "llama-mtmd-cli", dest)
    shutil.rmtree(src)
    log.unlink()


def ensure_runtime(allow_download: bool):
    """Locate (and, when allowed, build/download) the pinned llama.cpp
    binary and model weights. Sets the LLAMA_BIN global."""
    global LLAMA_BIN

    if os.environ.get("PENTACAM_OCR_LLAMA_BIN"):
        LLAMA_BIN = Path(os.environ["PENTACAM_OCR_LLAMA_BIN"])
    else:
        LLAMA_BIN = LLAMA_BIN_DEFAULT
        if not (LLAMA_BIN.exists() and _llama_commit_ok(LLAMA_BIN)):
            if not allow_download:
                sys.exit(f"Missing or wrong-version OCR binary: {LLAMA_BIN}")
            _build_llama(LLAMA_BIN)
            if not _llama_commit_ok(LLAMA_BIN):
                sys.exit(f"Built {LLAMA_BIN} but it does not report commit {LLAMA_COMMIT[:7]}.")

    for name, sha in MODEL_FILES.items():
        path = OCR_HOME / "models" / name
        # A marker records that this exact file was checksummed once, so
        # the ~4 GB aren't re-hashed on every run.
        marker = path.with_name(path.name + ".verified")
        if path.exists() and marker.exists() and marker.read_text().strip() == sha:
            continue
        if path.exists():
            print(f"Verifying {name} ...", flush=True)
            if _sha256(path) == sha:
                marker.write_text(sha)
                continue
            if not allow_download:
                break
            print(f"    {name} does not match the pinned version; re-downloading.")
            path.unlink()
        if not allow_download:
            break
        print(f"Downloading {name} ...", flush=True)
        _download(HF_URL + name, path, sha)
        marker.write_text(sha)

    missing = [str(p) for p in (LLAMA_BIN, MODEL, MMPROJ) if not p.exists()]
    if missing:
        sys.exit("Missing OCR runtime files: " + ", ".join(missing))


# ---------------------------------------------------------------------------
# Network isolation (patient data must never leave this machine)
# ---------------------------------------------------------------------------
#
# llama-mtmd-cli contains llama.cpp's built-in model downloader (HuggingFace /
# Docker registry via -hf, -mu, -dr or LLAMA_ARG_* env vars). We never use it,
# but rather than rely on that, on Linux the whole pipeline is re-executed
# inside a fresh network namespace (`unshare -rn`) where the only interface is
# a downed loopback -- the kernel refuses any connection attempt. If isolation
# can't be established (macOS/Windows, or Linux with unprivileged user
# namespaces disabled), the script refuses to run unless explicitly started
# with --no-network-isolation.

_ISOLATED_FLAG = "PENTACAM_OCR_NETNS_ISOLATED"

# Env vars that could make llama.cpp fetch a model or route traffic somewhere.
_BLOCKED_ENV_PREFIXES = ("LLAMA_ARG_", "LLAMA_CACHE", "HF_", "HUGGING")
_BLOCKED_ENV_NAMES = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}


def _network_interfaces() -> list:
    # /proc/self/net reflects this process's namespace (/sys/class/net does not).
    try:
        lines = Path("/proc/self/net/dev").read_text().splitlines()[2:]
    except OSError:
        return []
    return sorted(line.split(":")[0].strip() for line in lines)


def ensure_network_isolated(allow_unisolated: bool):
    """Re-exec this script inside an empty network namespace, then verify
    that no interface other than loopback is visible."""
    if os.environ.get(_ISOLATED_FLAG) != "1":
        unshare = shutil.which("unshare") if platform.system() == "Linux" else None
        if unshare:
            probe = subprocess.run([unshare, "-rn", "true"], capture_output=True, text=True)
            reason = probe.stderr.strip() if probe.returncode else None
        else:
            reason = "kernel network namespaces (`unshare`) are only available on Linux"
        if reason is None:
            env = dict(os.environ, **{_ISOLATED_FLAG: "1"})
            os.execve(unshare, [unshare, "-rn", sys.executable, os.path.abspath(__file__), *sys.argv[1:]], env)
        if allow_unisolated:
            print(f"WARNING: running WITHOUT network isolation ({reason}).\n"
                  f"         llama-mtmd-cli is only ever given local files, but nothing "
                  f"stops it from reaching the network.", file=sys.stderr)
            return
        sys.exit(
            f"Refusing to run: cannot guarantee network isolation ({reason}).\n"
            f"On Ubuntu 24.04+ this is usually AppArmor blocking unprivileged user\n"
            f"namespaces (sysctl kernel.apparmor_restrict_unprivileged_userns=0 re-enables\n"
            f"them). Alternatively disconnect from the network, or re-run with\n"
            f"--no-network-isolation to accept running without the guarantee."
        )

    ifaces = _network_interfaces()
    if ifaces != ["lo"]:
        sys.exit(f"Refusing to run: network namespace not isolated (interfaces: {ifaces}).")


def _ocr_env() -> dict:
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(_BLOCKED_ENV_PREFIXES) and k.lower() not in _BLOCKED_ENV_NAMES
    }
    # A shared-library build of llama-mtmd-cli keeps its .so files next to
    # the binary (harmless for the default static build).
    env["LD_LIBRARY_PATH"] = str(LLAMA_BIN.parent)
    return env


# ---------------------------------------------------------------------------
# OCR invocation
# ---------------------------------------------------------------------------

def run_ocr(image_path: Path, prompt: str = "Free OCR.") -> str:
    """Run llama-mtmd-cli on a single (small, cropped) image and return the
    generated text, stripped of the model's grounding/debug preamble."""
    cmd = [
        str(LLAMA_BIN),
        "-m", str(MODEL),
        "--mmproj", str(MMPROJ),
        "--image", str(image_path),
        "-p", prompt,
        "--chat-template", "deepseek-ocr",
        "--temp", "0",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=OCR_TIMEOUT,
        encoding="utf-8", errors="replace", env=_ocr_env(),
    )
    # The model's generated text goes to stdout; all loading/timing/debug
    # logging goes to stderr. (Concatenating the two loses chronological
    # order since they're captured as separate buffers, not interleaved
    # as they would be on a real terminal -- so stdout alone is correct.)
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Image cropping
# ---------------------------------------------------------------------------

def make_crop(src: Path, box, dst: Path):
    from PIL import Image
    img = Image.open(src)
    crop = img.crop(box)
    crop = crop.resize((crop.width * UPSCALE, crop.height * UPSCALE), Image.LANCZOS)
    crop.save(dst)


def detect_template(src: Path, workdir: Path) -> str:
    header_crop = workdir / f"{src.stem}_header.png"
    make_crop(src, TEMPLATE_REGIONS["belin_ambrosio"]["header"], header_crop)
    text = run_ocr(header_crop).lower()
    if "belin" in text or "ambrosio" in text or "ambr" in text or "ectasia" in text:
        return "belin_ambrosio"
    if "maps" in text:
        return "four_maps"
    return "refractive"


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------

LABEL_VALUE_RE = re.compile(
    r"[*\-\s#>]*\*{0,2}([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 .()/_\-]{0,40}?)\*{0,2}\s*:\s*"
    r"\*{0,2}([^\n|]+?)\*{0,2}\s*(?=(?:[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 .()/_\-]{0,40}?\s*:)|$)"
)

LABEL_ONLY_RE = re.compile(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 .()/_\-]{0,40}:\s*$")

HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """DeepSeek-OCR occasionally renders a dense number box as an HTML
    <table> instead of plain 'Label: value' lines. Worse, when it does
    this its own row/column alignment can be wrong (a label cell ends up
    paired with the NEXT row's value). Don't trust table structure at
    all -- just strip the tags so cell boundaries become plain
    whitespace/newlines for whatever parser runs next."""
    if "<table" not in text and "<tr>" not in text and "<td>" not in text:
        return text
    text = re.sub(r"</td>\s*<td>", "\n", text)
    text = re.sub(r"</tr>\s*<tr>", "\n", text)
    text = HTML_TAG_RE.sub("\n", text)
    return text


def _merge_label_only_lines(text: str) -> str:
    """DeepSeek-OCR sometimes puts a field's label and its value on two
    separate lines instead of 'Label: value' on one line (seen on the
    belin_ambrosio demographics box). Detect a line that is just 'Label:'
    with nothing after it, and splice the next non-blank line in as its
    value before the generic parser runs."""
    lines = [l.strip() for l in text.splitlines()]
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if LABEL_ONLY_RE.match(line):
            j = i + 1
            while j < len(lines) and not lines[j]:
                j += 1
            # A next line containing ':' is only another label if it
            # starts with a letter -- "13:14:12" is a Time value.
            if j < len(lines) and lines[j] and not re.match(r"[A-Za-zÀ-ÿ][^:]*:", lines[j]):
                out.append(f"{line} {lines[j]}")
                i = j + 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def parse_label_values(text: str) -> dict:
    """Generic 'Label: value' extractor. Handles multiple label:value pairs
    per line (as DeepSeek-OCR emits for two-column boxes), and disambiguates
    a label that repeats within the same region (e.g. 'Axis' appears once
    for K1 and once for K2) by numbering the repeats instead of silently
    overwriting the first one."""
    text = _strip_html(text)
    text = _merge_label_only_lines(text)
    fields = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for m in LABEL_VALUE_RE.finditer(line):
            label = m.group(1).strip().strip("*").strip()
            value = m.group(2).strip().strip("*").strip()
            if not label or not value:
                continue
            key = label
            n = 2
            while key in fields:
                key = f"{label} #{n}"
                n += 1
            fields[key] = value
    return fields


def parse_belin_k_readings(text: str) -> dict:
    """The belin_ambrosio K1/K2/KMax/Axis/Q-val/QS box. DeepSeek-OCR
    sometimes emits this as a malformed HTML table where the label column
    is shifted down one row from its value (K1's label ends up next to
    K2's value, etc.) -- so label-based parsing silently returns wrong
    numbers instead of failing loudly. Don't trust labels OR table
    position at all here: identify each value purely by its shape
    (D-suffixed diopter, deg-suffixed axis, bare decimal for Q-val) and
    rely on the fact that K1/K2/KMax always appear in that fixed order
    top-to-bottom on the real printout. A field whose expected token
    shape never appears is left out rather than guessed."""
    text = _strip_html(text)
    fields = {}

    # The model occasionally misreads the "D" suffix as "U" ("46.4U").
    diopters = re.findall(r"[-+]?\d+\.?\d*\s*[DU]\b", text)
    if len(diopters) == 3:
        for name, val in zip(["K1", "K2", "KMax"], diopters):
            fields[name] = re.sub(r"\s+", "", val)[:-1] + "D"
    else:
        # Fewer/more than three means the fixed-order assumption would
        # shift values onto the wrong K -- use the labels instead.
        for name in ["K1", "K2", "KMax"]:
            m = re.search(rf"\b{name}\s*:?\s*([-+]?\d+\.\d+)", text)
            if m:
                fields[name] = f"{m.group(1)}D"

    # "(30°)" / "[30°]" next to Q-val is fixed boilerplate (the zone the
    # Q-value is measured at), not a per-patient reading -- strip it
    # before hunting for the real Axis value so it can't be mistaken
    # for one.
    text_no_boilerplate = re.sub(r"[\[(]\s*30\s*°\s*[\])]", " ", text)
    degrees = re.findall(r"[-+]?\d+\.?\d*\s*°", text_no_boilerplate)
    if degrees:
        fields["Axis"] = degrees[0].strip()
    else:
        # The top of the K1/Axis row sits on the crop edge, so the "°" can
        # be clipped off -- fall back to a value explicitly labelled Axis.
        axis = re.search(r"Axis\.?\s*:?\s*([-+]?\d+\.?\d*)", text)
        if axis:
            fields["Axis"] = f"{axis.group(1)} °"

    remainder = text
    for consumed in diopters[:3] + degrees[:1]:
        remainder = remainder.replace(consumed, " ", 1)
    qval = re.search(r"[-+]\d+\.\d+", remainder)
    if qval:
        fields["Q-val"] = qval.group()

    qs = re.search(r"QS\.?:?\s*\n?\s*([A-Za-z][A-Za-z ]{1,20})", text)
    if qs:
        fields["QS"] = qs.group(1).strip()
    else:
        qs_fallback = re.search(r"\b(Data Gaps|Borderline|OK)\b", remainder)
        if qs_fallback:
            fields["QS"] = qs_fallback.group(1)

    return fields


def parse_belin_pachy_dist(text: str) -> dict:
    """The belin_ambrosio Pachy Thin. Locat. / Dist. Vertex box. The
    direction marker (IT, IN, ST, ...) printed between the Dist. label
    and its mm value lands wherever the model puts it -- often taken as
    the Dist. value itself, and sometimes both labels come out first with
    all values after them. As with parse_belin_k_readings, identify each
    value by shape instead: 3-digit um = thinnest pachy, decimal mm =
    distance, a lone I/S + T/N token = direction."""
    text = _strip_html(text)
    fields = {}
    for side in ("F", "B"):
        m = re.search(side + r"\.\s*Ele\.\s*Th\.?\s*:?\s*(\d+\s*[µμu]m)", text)
        if m:
            fields[f"{side}.Ele.Th"] = m.group(1)
            text = text.replace(m.group(0), " ", 1)
    pachy = re.search(r"\b\d{3}\s*[µμu]m", text)
    if pachy:
        fields["Pachy Thin. Locat."] = pachy.group()
    dist = re.search(r"\b\d+\.\d+\s*mm", text)
    if dist:
        fields["Dist. Vertex N.-Thin.Loc."] = dist.group()
    direction = re.search(r"(?m)^\W*([IS]?[TN]|[IS])\W*$", text)
    if direction:
        fields["Dist. Vertex N.-Thin.Loc. Direction"] = direction.group(1)
    return fields


PACHY_ROWS = [
    # "Pupil" is sometimes misread as "Pulp".
    (r"pu\w{1,3}\s*center", "Pachy Pupil Center", "Pupil Center"),
    (r"thinnest\s*locat", "Pachy Thinnest Locat", "Thinnest Locat"),
]
PACHY_NUM_RE = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?:\s*(?:µm|μm|um)\b)?")


def parse_pachy_table(text: str) -> dict:
    """The Pachy/pupil box has two table rows (Pupil Center, Thinnest
    Locat.) each with 3 numeric columns (value, x[mm], y[mm]) that the
    generic label:value parser can't separate. The model emits these
    as one row per line, a markdown table, or with every cell on its own
    line, so take the first three numbers after each row label (and
    before the next one) rather than relying on line structure."""
    fields = {}
    labels = []
    for pattern, pachy_key, xy_key in PACHY_ROWS:
        m = re.search(pattern + r"\.?\s*:?", text, re.I)
        if m:
            labels.append((m, pachy_key, xy_key))
    labels.sort(key=lambda t: t[0].start())
    consumed = []
    for i, (m, pachy_key, xy_key) in enumerate(labels):
        end = labels[i + 1][0].start() if i + 1 < len(labels) else len(text)
        nums = list(PACHY_NUM_RE.finditer(text, m.end(), end))[:3]
        consumed.append((m.start(), nums[-1].end() if len(nums) == 3 else m.end()))
        if len(nums) == 3:
            fields[pachy_key] = nums[0].group().strip()
            fields[f"{xy_key} x[mm]"] = nums[1].group().strip()
            fields[f"{xy_key} y[mm]"] = nums[2].group().strip()
    # Everything outside the two rows (A. C. Depth, Pupil Dia, Angle, ...)
    # is ordinary 'Label: value' text.
    rest = text
    for start, stop in reversed(consumed):
        rest = rest[:start] + "\n" + rest[stop:]
    rest = re.sub(r"(?m)^\s*(Pachy:|x\[?mm\]?|y\[?mm\]?)\s*$", "", rest)
    fields.update(parse_label_values(rest))
    return fields


FOUR_MAPS_PACHY_ROWS = {
    "pupil center": "Pupil Center",
    "pachy vertex": "Pachy Vertex",
    "thinnest locat": "Thinnest Locat",
    "k max": "KMax Front",
}


def _strip_markdown_table(text: str) -> str:
    """DeepSeek-OCR renders some 4 Maps boxes as a markdown table. Drop
    the |---| separator rows and turn cell borders into spaces so each
    row reads as plain 'Label: value' text."""
    lines = [l for l in text.splitlines() if not re.fullmatch(r"\s*\|?[\s|:\-]+\|?\s*", l)]
    return "\n".join(l.replace("|", " ") for l in lines)


def parse_four_maps_pachy(text: str) -> dict:
    """The 4 Maps Pachy box has four rows (Pupil Center, Pachy Vertex N.,
    Thinnest Locat., K Max. (Front)), each with value, x[mm], y[mm]. Each
    row's value cell is preceded by a marker glyph (+, dot, circle,
    diamond) -- the '+' is a symbol, not a sign, so numbers are only
    matched with a sign attached directly to the digits."""
    fields = {}
    for line in _strip_markdown_table(_strip_html(text)).splitlines():
        line = line.strip()
        for prefix, name in FOUR_MAPS_PACHY_ROWS.items():
            if line.lower().startswith(prefix):
                rest = line.split(":", 1)[1] if ":" in line else line[len(prefix):]
                nums = re.findall(r"[+-]?\d+\.?\d*(?:\s*(?:µm|um|D)\b)?", rest)
                if len(nums) >= 3:
                    fields[name] = nums[0].strip()
                    fields[f"{name} x[mm]"] = nums[1].strip()
                    fields[f"{name} y[mm]"] = nums[2].strip()
                break
    return fields


def _normalize_ocr_text(text: str) -> str:
    """Flatten the alternative layouts DeepSeek-OCR sometimes produces for
    the same box (seen on both the 4 Maps and Refractive templates) into
    plain one-field-per-line 'Label: value' text:
      - markdown tables  -> rows of space-separated cells
      - LaTeX values ("\\( 7.98 \\, \\text{mm} \\)") -> "7.98 mm"
      - bold bulleted lists with label and value on separate lines
      - several 'Label: value' pairs on one row"""
    text = _strip_markdown_table(_strip_html(text))
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[()\[\],]", " ", text)
    text = text.replace("**", "")
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)
    text = re.sub(r"(?m)^[ \t]+|[ \t]+$", "", text)
    # The 4 Maps "Enter IOP" button sits on the same row as IOP(Sum).
    text = re.sub(r"Enter\s+IOP\s*", "", text)
    # Fixed labels that wrap onto the value: "Axis: (steep)" and the
    # "Pachy:  x[mm]  y[mm]" column-header row.
    text = re.sub(r"[(\[]steep[)\]]\s*", "", text)
    # Values sometimes come back boxed in brackets: "[107.2 °]", "[OK]".
    text = re.sub(r"\[([^\[\]:]*)\]", r"\1", text)
    text = re.sub(r"(?m)^Pachy:\s*x\[?mm\]?\s*y\[?mm\]?\s*$", "", text)
    text = re.sub(r"\s{2,}(?=[A-Za-z][^:\n]{0,30}:)", "\n", text)
    return _merge_label_only_lines(text)


def parse_four_maps_region(region_name: str, text: str) -> dict:
    if region_name == "pachy":
        return parse_four_maps_pachy(text)
    region_fields = parse_label_values(_normalize_ocr_text(text))
    # Cornea Front and Cornea Back boxes repeat the same labels (Rf, K1,
    # Axis, ...), so namespace them by surface.
    surface = region_name.split("_")[0]
    if surface in ("front", "back"):
        region_fields = {f"{surface.title()} {k}": v for k, v in region_fields.items()}
    return {
        # Also drop a map-marker glyph read as part of the value ("◇ 5.59 mm").
        ("HWTW" if k.upper() == "HWTW" else k): re.sub(r"^[^\w+\-±.]+", "", re.sub(r"\s+", " ", v).strip("[] "))
        for k, v in region_fields.items()
    }


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def parse_region(template: str, region_name: str, text: str) -> dict:
    if template == "four_maps":
        return parse_four_maps_region(region_name, text)
    if region_name == "pachy_pupil":
        return parse_pachy_table(_normalize_ocr_text(text))
    if region_name == "k_readings" and template == "belin_ambrosio":
        return parse_belin_k_readings(text)
    if region_name == "pachy_dist":
        return parse_belin_pachy_dist(text)
    if region_name == "k_readings":
        region_fields = parse_label_values(text)
        # K1's and K2's "Axis" both land as separate numbered keys
        # ("Axis", "Axis #2") since the label repeats in this box;
        # give them their real clinical meaning instead of a number.
        if "Axis" in region_fields:
            region_fields["K1 Axis"] = region_fields.pop("Axis")
        if "Axis #2" in region_fields:
            region_fields["K2 Axis"] = region_fields.pop("Axis #2")
        return region_fields
    # The model sometimes repeats a whole box verbatim; drop "Label #2"
    # copies that just echo the first occurrence's value.
    region_fields = parse_label_values(text)
    return {
        k: v for k, v in region_fields.items()
        if not (re.search(r" #\d+$", k) and region_fields.get(re.sub(r" #\d+$", "", k)) == v)
    }


def process_image(src: Path, workdir: Path, raw_dir: Path) -> dict:
    template = detect_template(src, workdir)
    regions = TEMPLATE_REGIONS[template]

    record = {"file": src.name, "template": template}
    raw_chunks = [f"# {src.name} ({template})\n"]

    for region_name, box in regions.items():
        if region_name == "header":
            continue
        crop_path = workdir / f"{src.stem}_{region_name}.png"
        make_crop(src, box, crop_path)
        text = run_ocr(crop_path)
        raw_chunks.append(f"## {region_name}\n{text}\n")

        record.update(parse_region(template, region_name, text))

    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{src.stem}.md").write_text("\n".join(raw_chunks), encoding="utf-8")

    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, nargs="?", help="Folder of Pentacam JPG printouts")
    ap.add_argument("--out", type=Path, default=None,
                     help="Output basename (default: <folder>/ocr_results)")
    ap.add_argument("--pattern", default="*.JPG", help="Glob pattern for images")
    ap.add_argument("--setup-only", action="store_true",
                    help="Download/verify the OCR runtime and model, then exit")
    ap.add_argument("--no-network-isolation", action="store_true",
                    help="Allow running where kernel network isolation is unavailable "
                         "(macOS, Windows, restricted Linux)")
    args = ap.parse_args()
    if args.folder is None and not args.setup_only:
        ap.error("folder is required (or use --setup-only)")

    try:
        import PIL  # noqa: F401
    except ImportError:
        sys.exit("Pillow is required: pip install pillow")

    # Downloads only ever happen here, in the original (networked) process,
    # before any image is read. The isolated re-exec below re-runs this with
    # downloads disabled, so it just confirms the files are present.
    isolated = os.environ.get(_ISOLATED_FLAG) == "1"
    ensure_runtime(allow_download=not isolated)
    if args.setup_only:
        print(f"OCR runtime ready:\n  {LLAMA_BIN}\n  {MODEL}\n  {MMPROJ}")
        return

    ensure_network_isolated(args.no_network_isolation)

    images = sorted(args.folder.glob(args.pattern))
    if not images:
        images = sorted(args.folder.glob("*.jpg"))
    if not images:
        sys.exit(f"No images found in {args.folder} matching {args.pattern}")

    out_base = args.out or (args.folder / "ocr_results")
    workdir = args.folder / ".ocr_crops"
    raw_dir = args.folder / "ocr_raw_text"
    workdir.mkdir(exist_ok=True)

    records = []
    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img.name} ...", flush=True)
        try:
            record = process_image(img, workdir, raw_dir)
            records.append(record)
            print(f"    -> {len(record) - 2} fields extracted")
        except Exception as e:
            print(f"    !! FAILED: {e}", file=sys.stderr)
            records.append({"file": img.name, "template": "ERROR", "error": str(e)})

    # JSON (nested, keeps every field per image)
    json_path = out_base.with_suffix(".json")
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV (flat, union of all field names seen)
    all_keys = []
    for r in records:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    csv_path = out_base.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Raw per-region OCR text: {raw_dir}/")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Setup notes
# ---------------------------------------------------------------------------
#
# Beyond the requirements in the module docstring, ensure_runtime() does
# everything: it compiles LLAMA_COMMIT and downloads the MODEL_FILES listed
# in CONFIG. Changing either changes the OCR output, so re-check the
# extracted fields against the printouts after any change.
#
# Environment variables:
#   PENTACAM_OCR_HOME      where the runtime is stored (default ~/.local/share/deepseek-ocr)
#   PENTACAM_OCR_LLAMA_BIN use this llama-mtmd-cli instead of building the pinned commit
#   PENTACAM_OCR_TIMEOUT   seconds allowed per OCR call (default 900)
#
# Offline machines: run `pentacam_ocr.py --setup-only` on a networked
# machine, then copy the PENTACAM_OCR_HOME folder across.

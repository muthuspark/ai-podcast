#!/usr/bin/env python3
"""Vet a downloaded head model before using it for lip-sync.

A model is USABLE only if it has morph targets (blendshapes) AND one of them
opens the mouth (jawOpen / mouthOpen / a viseme). Sketchfab GLB exports often
strip morph targets even when the page claims blendshapes — this catches that.

    python3 tools/check_glb.py static/*.glb
"""
import json
import struct
import sys

# substrings that identify a "mouth opens" morph, across naming conventions
MOUTH_KEYS = ("jawopen", "mouthopen", "viseme", "mouth_open", "aa", "_open")
# the morphs our app actually drives — a model with these animates richly
NICE = ("jawOpen", "eyeBlink_L", "eyeBlink_R", "mouthSmile_L", "mouthSmile_R")


def load_gltf_json(path):
    with open(path, "rb") as f:
        head = f.read(16)
        f.seek(0)
        data = f.read()
    if data[:4] == b"glTF":                       # binary .glb: JSON is first chunk
        jlen = struct.unpack("<I", data[12:16])[0]
        return json.loads(data[20:20 + jlen])
    return json.loads(data.decode("utf-8"))        # plain .gltf


def morph_names(j):
    names = []
    for m in j.get("meshes", []):
        m_extras = (m.get("extras") or {}).get("targetNames") or []
        for p in m.get("primitives", []):
            p_extras = (p.get("extras") or {}).get("targetNames") or []
            n = len(p.get("targets", []))
            names += (m_extras or p_extras or [f"morph{i}" for i in range(n)])
    return names


def check(path):
    try:
        j = load_gltf_json(path)
    except Exception as e:
        return False, f"not a readable glTF/glb ({e})"
    req = j.get("extensionsRequired") or []
    unsupported = [x for x in req if x not in (
        "KHR_mesh_quantization", "EXT_meshopt_compression",
        "KHR_texture_basisu", "KHR_texture_transform", "KHR_materials_unlit")]
    names = morph_names(j)
    if not names:
        return False, "0 morph targets — will NOT lip-sync (the common Sketchfab trap)"
    mouth = [x for x in names if any(k in x.lower() for k in MOUTH_KEYS)]
    if not mouth:
        return False, f"{len(names)} morphs but none opens the mouth: {names[:8]}…"
    extra = f"  ⚠ needs unsupported ext {unsupported}" if unsupported else ""
    nice = sum(1 for k in NICE if k in names)
    return True, f"{len(names)} morphs, mouth={mouth[0]}, ARKit-match {nice}/{len(NICE)}{extra}"


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    bad = 0
    for path in argv:
        ok, msg = check(path)
        print(f"{'PASS' if ok else 'FAIL'}  {path}\n      {msg}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

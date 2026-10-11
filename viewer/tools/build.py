"""Build only allowlisted artifacts from the pinned, publicly released commit.

No model imports, network calls, training, or reads of ignored experiment files.
Run from any directory: python viewer/tools/build.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

VIEWER = Path(__file__).resolve().parents[1]
ROOT = VIEWER.parent
OUT = VIEWER / "dist"
PUBLIC_COMMIT = "74e8f7be101e050c6982f2838f2d3ea980851f5f"
REPO = "https://github.com/mihir-s-05/icl-selfplay-vs-text"
TABLES = ("headline", "tasks", "mcurves", "valbpb", "sp_bpb")
FIGURES = ("fig1_scaling", "fig2_tokens", "fig3_tasks", "fig4_warm", "fig5_mcurves")
EXPECTED_ROWS = {"headline": 1191, "tasks": 8337, "mcurves": 78209, "valbpb": 1248, "sp_bpb": 198}
NUMERIC = {"tokens", "value", "val_bpb", "dclm_bpb", "m", "round"}


def released(path: str) -> bytes:
    return subprocess.check_output([
        "git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT),
        "show", f"{PUBLIC_COMMIT}:{path}",
    ])


def write(path: str, content: bytes) -> None:
    target = OUT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def main() -> None:
    manifest = {"schemaVersion": 1, "sourceCommit": PUBLIC_COMMIT,
                "repository": REPO, "license": "Apache-2.0", "tables": {}}
    for name in TABLES:
        source = f"writeup/figs/data/{name}.csv"
        raw = released(source)
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
        columns = reader.fieldnames
        assert columns is not None
        rows = list(reader)
        assert len(rows) == EXPECTED_ROWS[name], f"Unexpected row count: {name}"
        value_key = next(k for k in columns if k in {"value", "val_bpb", "dclm_bpb"})
        keys = [tuple(r[k] for k in columns if k != value_key) for r in rows]
        assert len(set(keys)) == len(keys), f"Duplicate measurements: {name}"
        for row in rows:
            for key in NUMERIC.intersection(columns):
                row[key] = float(row[key]) if key not in {"m", "round"} else int(row[key])
            if value_key == "value":
                assert 0 <= row[value_key] <= 1
        # Dictionary-encoded columns keep all measurements, including exact seed IDs.
        dictionaries = {k: sorted({r[k] for r in rows}) for k in columns if k not in NUMERIC}
        indices = {k: {v: i for i, v in enumerate(values)} for k, values in dictionaries.items()}
        encoded = [[indices[k][r[k]] if k in indices else r[k] for k in columns] for r in rows]
        payload = {"columns": columns, "dictionaries": dictionaries, "rows": encoded}
        write(f"data/{name}.json", json.dumps(payload, separators=(",", ":")).encode())
        write(f"data/{name}.csv", raw)
        manifest["tables"][name] = {"rows": len(rows), "columns": columns,
            "source": f"{REPO}/blob/{PUBLIC_COMMIT}/{source}",
            "sha256": hashlib.sha256(raw).hexdigest(), "csvBytes": len(raw),
            "tokensUnit": "billions of byte-level learner tokens"}
    for name in FIGURES:
        for extension in ("svg", "pdf"):
            write(f"figures/{name}.{extension}", released(f"writeup/figs/{name}.{extension}"))
    write("data/writeup.md", released("writeup/WRITEUP.md"))
    write("LICENSE", released("LICENSE"))
    write("NOTICE", released("NOTICE"))
    for source in sorted((VIEWER / "src").iterdir()):
        if source.is_file():
            shutil.copyfile(source, OUT / source.name)
    write("data/manifest.json", json.dumps(manifest, indent=2).encode())
    print(f"Built {OUT}: {sum(EXPECTED_ROWS.values()):,} released rows; no model dependencies.")


if __name__ == "__main__":
    main()

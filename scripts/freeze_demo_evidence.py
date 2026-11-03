"""Create the trusted checksum list after curating real evidence; never run during replay."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root",type=Path,default=Path("results"))
    parser.add_argument("--output",type=Path,default=Path("src/metalmind/demonstration/evidence-manifest.json"))
    args=parser.parse_args();root=args.evidence_root.resolve();files={}
    for name in ("2026-11-02-fp32-pilot","2026-11-02-module-optimizer","2026-11-02-module-baselines"):
        for path in sorted((root/name).rglob("*")):
            if path.is_file() and path.suffix in (".json",".jsonl",".metal",".py"):
                if not path.resolve().is_relative_to(root): raise ValueError("Evidence file escapes its root")
                files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not files: raise ValueError("No evidence files found")
    args.output.write_text(json.dumps({"schema_version":1,"files":files},sort_keys=True,indent=2)+"\n")
    print("Frozen evidence checksums:",len(files))


if __name__=="__main__": main()

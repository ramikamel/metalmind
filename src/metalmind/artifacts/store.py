"""Atomic manifests and immutable source snapshots; no credentials stored."""
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,indent=2,allow_nan=False)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix=".pending-")
    try:
        with os.fdopen(fd,"w") as f: f.write(encoded);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)

class ArtifactStore:
    def __init__(self,path):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)

    @classmethod
    def create_run(cls,root,config,environment):
        from dataclasses import asdict
        store=cls(Path(root)/uuid.uuid4().hex)
        store.write_manifest({"config":config.model_dump(mode="json"),"environment":asdict(environment),"phase":"created"})
        return store

    def write_source(self,candidate_id,source):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+",candidate_id): raise ValueError("invalid source id")
        p=self.path/"candidates"/(candidate_id+".metal");p.parent.mkdir(exist_ok=True)
        try:
            with p.open("x") as f: f.write(source)
        except FileExistsError:
            if p.read_text()!=source: raise ValueError("candidate source is immutable") from None
        return p

    def append_event(self,event):
        p=self.path/"events.jsonl"
        if p.exists():
            data=p.read_bytes()
            if data and not data.endswith(b"\n"):
                with p.open("wb") as f: f.write(data[:data.rfind(b"\n")+1])
                self.append_event({"event":"recovered_truncated_tail"})
        with p.open("a") as f:
            f.write(json.dumps(event,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())

    def read_events(self):
        p=self.path/"events.jsonl"
        if not p.exists(): return []
        data=p.read_bytes(); lines=data.splitlines(keepends=True)
        return [json.loads(line) for line in lines if line.endswith(b"\n")]

    def write_manifest(self,value): atomic_json(self.path/"manifest.json",value)
    def read_manifest(self): return json.loads((self.path/"manifest.json").read_text())

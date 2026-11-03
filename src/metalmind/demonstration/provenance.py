"""Identify actual package bytes independently of a checkout's working-tree state."""
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess


def _digest(files):
    digest=hashlib.sha256();count=0
    for name,data in sorted(files):
        digest.update(name.encode()+b"\0"+data+b"\0");count+=1
    return {"sha256":digest.hexdigest(),"file_count":count}


def package_fingerprint(root):
    root=Path(root).resolve()
    return _digest((p.relative_to(root).as_posix(),p.read_bytes()) for p in root.rglob("*") if p.is_file() and p.suffix in (".py",".json"))


def git_package_fingerprint(checkout):
    names=subprocess.check_output(["git","ls-tree","-r","--name-only","HEAD","src/metalmind"],cwd=checkout,text=True).splitlines()
    return _digest((n.removeprefix("src/metalmind/"),subprocess.check_output(["git","show","HEAD:"+n],cwd=checkout)) for n in names if Path(n).suffix in (".py",".json"))


def validate_runtime_source(actual,expected):
    if actual.get("sha256")!=expected["sha256"] or actual.get("file_count")!=expected["file_count"] or not expected["file_count"]:
        raise ValueError("Executed package source differs from the recorded checkout revision; install the committed checkout")


def runtime_source_info():
    import metalmind
    root=Path(metalmind.__file__).resolve().parent
    return {**package_fingerprint(root),"package_location":str(root),"package_version":importlib.metadata.version("metalmind")}

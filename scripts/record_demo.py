"""Record actual offline CLI output; export no secrets and make no model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root",type=Path,default=Path("results"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--gpu-check",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    folder=args.output_dir.resolve();folder.mkdir(parents=True,exist_ok=False)
    evidence=args.evidence_root.resolve()
    env=dict(os.environ)
    for key in ("MODEL_API_KEY","PYTHONPATH"): env.pop(key,None)
    argv=[sys.executable,"-m","metalmind.cli","demo","--evidence-root",str(evidence),"--output",str(folder/"demo.json"),"--pace","0.08"]
    if args.gpu_check: argv.append("--gpu-check")
    from metalmind.demonstration.provenance import git_package_fingerprint,validate_runtime_source
    expected=git_package_fingerprint(root)
    script_head=subprocess.check_output(["git","show","HEAD:scripts/record_demo.py"],cwd=root)
    if script_head!=Path(__file__).read_bytes(): raise ValueError("Commit recorder changes before creating a reproducible recording")
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix="metalmind-demo-recording-") as scratch:
        assert not (Path(scratch)/".env").exists()
        probe="import json; from metalmind.demonstration.provenance import runtime_source_info; print(json.dumps(runtime_source_info()))"
        executed=json.loads(subprocess.check_output([sys.executable,"-c",probe],cwd=scratch,env=env,text=True))
        validate_runtime_source(executed,expected)
        with (folder/"demo.cast").open("w") as cast,(folder/"transcript.txt").open("w") as transcript:
            header={"version":2,"width":100,"height":32,"timestamp":int(time.time()),"title":"MetalMind: saved evidence replay; no model generation","env":{"TERM":"xterm-256color"}}
            cast.write(json.dumps(header)+"\n")
            process=subprocess.Popen(argv,cwd=scratch,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                transcript.write(line);cast.write(json.dumps([round(time.monotonic()-started,4),"o",line])+"\n")
            returncode=process.wait()
        if returncode: raise SystemExit("CLI recording failed; inspect its saved transcript")
    data=json.loads((folder/"demo.json").read_text());assert data["model_requests"]==0
    validate_runtime_source(data["runtime_source"],expected)
    context={"schema_version":1,"mode":"actual CLI output recording","model_key_absent":True,"local_env_file_absent":True,
        "working_directory_outside_checkout":True,"model_requests":0,"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
        "executed_package":executed,"source_matches_commit":True,"recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command":argv,"recording_seconds":time.monotonic()-started,
        "sha256":{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ("demo.cast","demo.json","transcript.txt")}}
    (folder/"capture.json").write_text(json.dumps(context,indent=2)+"\n")
    print(json.dumps({"output_dir":str(folder),"live_check":data["live_check"],"chapters":len(data["chapters"]),"model_requests":0},indent=2))


if __name__=="__main__": main()

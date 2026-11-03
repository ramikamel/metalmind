import json
import textwrap
import time
from metalmind.artifacts.store import atomic_json
from metalmind.demonstration.evidence import load_demo_evidence,child,PILOT
from metalmind.demonstration.story import build_chapters,safe_text


def run_demo(args):
    if not 0<=args.pace<=2: raise ValueError("Presentation pace must be between 0 and 2 seconds per line")
    data=load_demo_evidence(args.evidence_root)
    if args.gpu_check:
        from metalmind.demonstration.checks import check_artifact_reuse
        data["live_check"]=check_artifact_reuse(child(args.evidence_root,PILOT,"artifacts",data["selected_artifact_id"]))
    from metalmind.demonstration.provenance import runtime_source_info
    data["runtime_source"]=runtime_source_info()
    data["chapters"]=build_chapters(data)
    if args.output: atomic_json(args.output,data)
    if args.json: print(json.dumps(data,indent=2,allow_nan=False))
    else:
        for index,chapter in enumerate(data["chapters"],1):
            lines=[f"[{index}/{len(data['chapters'])}] {chapter['title']}",chapter['label'],""]
            lines.extend(line for paragraph in chapter["lines"] for line in textwrap.wrap(paragraph,width=96))
            for line in lines+[""]:
                print(safe_text(line),flush=True)
                if args.pace: time.sleep(args.pace)
    return 1 if (data.get("live_check") or {}).get("outcome")=="failed" else 0

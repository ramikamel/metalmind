"""Render a paced evidence walkthrough, not a live search or desktop screencast.

Requires pip install -e '.[demo]'. No model generation or network assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import textwrap
from PIL import Image,ImageDraw,ImageFont,ImageStat
import PIL
import imageio_ffmpeg

WIDTH,HEIGHT=1920,1080
BG="#0b1119";PANEL="#111d29";FG="#ecf5f7";MUTED="#9cafbb";ACCENT="#62e3bd"


def font(size,mono=False):
    candidates=([Path("/System/Library/Fonts/Menlo.ttc"),Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")] if mono else
        [Path("/System/Library/Fonts/Supplemental/Arial.ttf"),Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")])
    for p in candidates:
        if p.is_file(): return ImageFont.truetype(str(p),size=size)
    return ImageFont.load_default(size=size)


def wrap(text,face,width):
    lines=[]
    for paragraph in text.splitlines():
        current=""
        for word in paragraph.split(" "):
            proposal=(current+" "+word).lstrip()
            if face.getlength(proposal)>width and current:
                lines.append(current);current=word
            else: current=proposal
            while face.getlength(current)>width:
                split=max(1,int(len(current)*width/face.getlength(current)))
                lines.append(current[:split]);current=current[split:]
        lines.append(current)
    return lines


def frame(chapter,index,total,folder,scope):
    image=Image.new("RGB",(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((70,46,1850,104),radius=18,fill=PANEL)
    draw.text((96,61),"METALMIND  /  VERIFIED GPU OPTIMIZATION",font=font(23,True),fill=ACCENT)
    draw.text((1685,61),f"{index+1:02d} / {total:02d}",font=font(23,True),fill=MUTED)
    draw.text((96,144),chapter["title"],font=font(64),fill=FG)
    draw.text((100,238),chapter["label"],font=font(24,True),fill=ACCENT)
    draw.rounded_rectangle((70,300,1850,942),radius=24,fill=PANEL)
    y=342
    size=32 if "code" not in chapter else 27
    paragraphs=list(chapter["lines"])
    while size>20:
        face=font(size,True);lines=[line for p in paragraphs for line in wrap(p,face,1660)]+([""] if "code" in chapter else [])
        if len(lines)*(size+17)+(0 if "code" not in chapter else 350)<=560: break
        size-=1
    face=font(size,True)
    for p in paragraphs:
        for line in wrap(p,face,1660):
            draw.text((104,y),line,font=face,fill=FG);y+=size+17
        y+=14
    if "code" in chapter:
        codefont=font(22,True)
        for line in wrap(chapter["code"],codefont,1660):
            if y>902:
                draw.text((104,y),"... (source excerpt; full verified kernel in the repository)",font=codefont,fill=MUTED);break
            draw.text((104,y),line,font=codefont,fill=MUTED);y+=29
    if y>930: raise ValueError("Demo chapter exceeds its readable text area: "+chapter["title"])
    draw.text((96,983),scope,font=font(22,True),fill=MUTED)
    draw.text((96,1020),"Presentation pacing is not benchmark timing. No live model generation.",font=font(21,True),fill=MUTED)
    path=folder/f"frame-{index:02d}.png";image.save(path)
    return path


def timestamp(seconds):
    value=int(seconds*1000);hours,value=divmod(value,3600000);minutes,value=divmod(value,60000);secs,ms=divmod(value,1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seconds-per-chapter",type=int,default=14)
    args=parser.parse_args()
    if not 5<=args.seconds_per_chapter<=30: parser.error("Chapter duration must be 5–30 seconds")
    source=args.recording_dir.resolve();folder=args.output_dir.resolve();folder.mkdir(parents=True,exist_ok=False)
    data=json.loads((source/"demo.json").read_text());capture=json.loads((source/"capture.json").read_text())
    if data["model_requests"]!=0 or capture["sha256"]["demo.json"]!=hashlib.sha256((source/"demo.json").read_bytes()).hexdigest():
        raise ValueError("Recording payload or zero-request provenance differs")
    chapters=data["chapters"];frames=[];captions=["WEBVTT",""];voice=[]
    modes=", ".join(sorted({w["domain_mode"] for w in data["function_winners"]}))
    scope=f"SAVED PERFORMANCE EVIDENCE  |  domain: {modes}  |  {data['hardware']['model']}, {data['hardware']['cores']} GPU cores"
    for i,c in enumerate(chapters):
        frames.append(frame(c,i,len(chapters),folder,scope))
        start=i*args.seconds_per_chapter;end=(i+1)*args.seconds_per_chapter
        captions.extend([str(i+1),timestamp(start)+" --> "+timestamp(end),c["title"]," ".join(c["lines"]),""])
        voice.extend([f"## {timestamp(start)} — {c['title']}"," ".join(c["lines"]),""])
    concat=[]
    for p in frames: concat.extend(["file '"+p.name+"'","duration "+str(args.seconds_per_chapter)])
    concat.append("file '"+frames[-1].name+"'")
    (folder/"concat.txt").write_text("\n".join(concat)+"\n")
    (folder/"captions.vtt").write_text("\n".join(captions)+"\n")
    (folder/"voiceover.md").write_text("# Optional narration script\n\nSilent video; record your own narration if desired.\n\n"+"\n".join(voice))
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();video=folder/"metalmind-evidence-walkthrough.mp4"
    command=[ffmpeg,"-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",str(folder/"concat.txt"),
        "-t",str(len(chapters)*args.seconds_per_chapter),"-r","24","-c:v","libx264","-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart","-an",str(video)]
    subprocess.run(command,cwd=folder,check=True)
    reader=imageio_ffmpeg.read_frames(str(video));metadata=next(reader);decoded=Image.frombytes("RGB",metadata["size"],next(reader));reader.close()
    if metadata["size"]!=(WIDTH,HEIGHT) or abs(metadata["duration"]-len(chapters)*args.seconds_per_chapter)>.5 or max(ImageStat.Stat(decoded).stddev)<10:
        raise ValueError("Rendered video metadata/frame verification failed")
    shutil.copy2(frames[0],folder/"poster.png")
    report={"schema_version":1,"mode":"paced saved-evidence walkthrough, not a desktop screencast or live generation",
        "silent":True,"model_requests":0,"chapter_count":len(chapters),"duration_seconds":metadata["duration"],"size":metadata["size"],"fps":metadata["fps"],
        "source_demo_sha256":hashlib.sha256((source/"demo.json").read_bytes()).hexdigest(),"source_commit":capture["source_commit"],
        "pillow_version":PIL.__version__,"imageio_ffmpeg_version":imageio_ffmpeg.__version__,
        "sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (video,folder/"captions.vtt",folder/"poster.png")}}
    (folder/"render.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()

from flask import (
    Flask,
    jsonify,
    request,
    Response,
    render_template_string,
    abort,
    send_from_directory,
    send_file,
)
from flask_cors import CORS, cross_origin

import modules.wav2lip.wav2lip_module as wav2lip_module
import json, os, random
from types import SimpleNamespace


parent_dir = os.path.dirname(os.path.abspath(__file__))

# Used for making text xml compatible, needed for voice pitch and speed control
table = str.maketrans({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
})

def xmlesc(txt):
    return txt.translate(table)
    

print("Initializing wav2lip module")


wav2lip_args_json = '''{
    "checkpoint_path": "modules/wav2lip/checkpoints/wav2lip.pth", 
    "face": "modules/wav2lip/input/default/", 
    "audio":"test.wav", 
    "outfile":"modules/wav2lip/output/wav2lip.mp4", 
    "img_size":96, 
    "fps":15, 
    "wav2lip_batch_size":1024, 
    "box":[-1, -1, -1, -1], 
    "face_det_batch_size":16,
    "pads":[0, 10, 0, 0],
    "crop":[0, -1, 0, -1], 
    "nosmooth": "False", 
    "resize_factor":1, 
    "rotate":"False"}'''
wav2lip_args = json.loads(wav2lip_args_json, object_hook=lambda d: SimpleNamespace(**d))



# generate and save video, returns nothing
def wav2lip_server_generate(fname="wav2lip"):
    files = [ f for f in os.listdir("modules/wav2lip/input/default/") if os.path.isfile(os.path.join("modules/wav2lip/input/default/",f)) ]
    rand_r = random.randrange(0, len(files))
    print("wav2lip starting with input: "+files[rand_r])
    wav2lip_args.face = "modules/wav2lip/input/default/"+files[rand_r]
    wav2lip_args.outfile = "modules/wav2lip/output/"+fname+".mp4"
    wav2lip_module.wav2lip_main(wav2lip_args)
    return "no"


# return created video
WAV2LIP_OUTPUT_PATH = os.path.join(parent_dir, "output\\")
def wav2lip_server_play(fname):
    print(WAV2LIP_OUTPUT_PATH)
    print(fname)
    return send_from_directory(WAV2LIP_OUTPUT_PATH, f"{fname}.mp4")
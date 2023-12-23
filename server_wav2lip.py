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
    "rotate":"False",
    "device":"cpu"}'''
wav2lip_args = json.loads(wav2lip_args_json, object_hook=lambda d: SimpleNamespace(**d))



# generate and save video, returns nothing
def wav2lip_server_generate(char_folder="default", device="cpu", audio="test"):
    files = [ f for f in os.listdir("modules/wav2lip/input/"+char_folder+"/") if os.path.isfile(os.path.join("modules/wav2lip/input/"+char_folder+"/",f)) ]
    rand_r = random.randrange(0, len(files))
    print("wav2lip starting with input: "+files[rand_r])
    wav2lip_args.face = "modules/wav2lip/input/"+char_folder+"/"+files[rand_r]
    wav2lip_args.outfile = "modules/wav2lip/output/wav2lip.mp4"
    wav2lip_args.device = device
    wav2lip_args.audio = audio+".wav" # test.wav from silero and out.wav from xttsv2
    wav2lip_module.wav2lip_main(wav2lip_args)
   
    return "True"


# return created video

def wav2lip_server_play(fname, char_folder):
    if fname == "silence":
        WAV2LIP_OUTPUT_PATH = os.path.join(parent_dir, "input\\"+char_folder+"\\")
    else:
        WAV2LIP_OUTPUT_PATH = os.path.join(parent_dir, "output\\")    
    print(WAV2LIP_OUTPUT_PATH)
    print(fname)
    return send_from_directory(WAV2LIP_OUTPUT_PATH, f"{fname}.mp4")
    
# SET and load silero language model
def wav2lip_server_silero_set_lang(tts_service, fname):
    tts_service.load_model(lang_model=fname+".pt")
    print("silero model "+fname+".pt loaded ")
    return "True"

# GET char folders in input folder
def wav2lip_server_get_chars():
    folders = [ f for f in os.listdir("modules/wav2lip/input/") if os.path.isdir(os.path.join("modules/wav2lip/input/",f)) ]
    return folders
    
# silero tts generate (override)
def wav2lip_server_tts_generate(tts_service, voice):
    print("in wav2lip_server_tts_generate")
    print(voice)
    if "text" not in voice or not isinstance(voice["text"], str):
        abort(400, '"text" is required')
    if "speaker" not in voice or not isinstance(voice["speaker"], str):
        abort(400, '"speaker" is required')
    if "voice_speed" not in voice or not isinstance(voice["voice_speed"], str):
        voice['voice_speed'] = 'medium'
    if "voice_pitch" not in voice or not isinstance(voice["voice_pitch"], str):
        voice['voice_pitch'] = 'medium'
        
    # Remove asterisks
    voice['text'] = voice['text'].replace("*", "")
    try:
        # Remove the destination file if it already exists
        #if os.path.exists('../../test.wav'):
        #    os.remove('../../test.wav')
        prosody = '<prosody rate="{}" pitch="{}">'.format(voice['voice_speed'], voice['voice_pitch'])
        silero_input = '<speak>'+prosody+xmlesc(voice['text'])+'</prosody></speak>'
        audio = tts_service.generate(voice['speaker'], silero_input)
        #audio_file_path = os.path.join(os.path.dirname(os.path.abspath(audio)), os.path.basename(audio))
        audio_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"+os.path.basename(audio))
        #audio_file_path = +audio_file_path
        print("audio: "+str(audio))
        print("audio_file_path: "+str(audio_file_path))
        #os.rename(audio, audio_file_path)
        return send_file(audio_file_path, mimetype="audio/x-wav")
    except Exception as e:
        print(e)
        abort(500, voice['speaker'])
        
    return "True"    
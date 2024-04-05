from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, modules.wav2lip.audio as audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, modules.wav2lip.face_detection as face_detection
import torch.nn.functional as F
from modules.wav2lip.models import Wav2Lip
import platform, time
import pickle
from huggingface_hub import hf_hub_download
from pathlib import Path
import ffpyplayer
import signal

# future work
#import cupy as cp
# cupy failed to 'pip install cupy' on my system, check realeses manually https://github.com/cupy/cupy/releases
# e.g. pip install https://github.com/cupy/cupy/releases/download/v13.0.0/cupy_cuda11x-13.0.0-cp311-cp311-win_amd64.whl

# globals
model = None                # to keep model in vram between runs
mel_step_size = 16          
device = '' # cpu, cuda     # use request param to set device
full_frames_by_file = {}    #
start_frame_global = 0             # start frame to play from current video, to make smooth chunks playing

'''
from functools import wraps
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
from flask_cors import CORS
from flask_compress import Compress
'''

'''
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
                    help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                                default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
                    help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=1024)

parser.add_argument('--resize_factor', default=1, type=int, 
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96
'''

'''
app = Flask(__name__)
parent_dir = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_OUTPUT_PATH = os.path.join(parent_dir, "tts_samples")
app = Flask(__name__)

@app.route("/api/wav2lip/play/<fname>", methods=["GET"])
def wav2lip_play_video(fname: str):
    return send_from_directory(WAV2LIP_OUTPUT_PATH, f"{fname}.mp4")
'''

def kill_flask_server():
        os.kill(os.getpid(), signal.SIGINT)
        return jsonify({ "success": True, "message": "Server is shutting down..." })
        
def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, args):
    global face_detect_running, device
    
    os.environ["face_detect_running"] = "1"
    print ("face_detect is running on device:"+device)
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device="cuda:0")

    face_det_batch_size = args.face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), face_det_batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + face_det_batch_size])))
        except RuntimeError:
            if face_det_batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            face_det_batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(face_det_batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            #cv2.imwrite(f"{Path(__file__).parent}/temp/faulty_frame.jpg", image) # check this frame where the face was not detected.
            y1 = x1 = 0
            x2 = y2 = 1
            #print ('face not found in frame '+str(len(results))+'. skipping it')
            #raise ValueError('Face not detected in frame '+str(len(results))+'/'+str(len(images))+'. frame saved to temp/faulty_frame.jpg. Ensure the video contains a face in all the frames.')
        else:
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    detector.unload()
    del detector
    del predictions
    torch.cuda.empty_cache()
    os.environ["face_detect_running"] = "0"
    
    return results 

def face_detect_with_cache(images, args, input_video_fname, input_file_size):
    global device
    # Check if cache exists
    cache_filename = input_video_fname+"_"+str(input_file_size)+".pkl"
    cache_path = f"{Path(__file__).parent}/cache/face_detection/{cache_filename}"
    if not Path(cache_path).exists():
        print("Face detection cache "+cache_filename+" doesn't exist, calling face detect using device: "+device)
        print("***")
        print("There's a bug currently (memory leak), please restart 'Silly Tavern Extras' after Face detection process is finished. It is needed just once for each new video.")
        print("***")
        faces = face_detect(images, args)
               
        # Save encoded faces to cache file
        with open(cache_path, "wb") as f:
            pickle.dump(faces, f)
            
        print("Please restart 'Silly Tavern Extras' (this window) now.")
        kill_flask_server()    
    
    else:
        # Load cached faces from file
        print("Loading cached faces from file "+cache_filename)
        with open(cache_path, "rb") as f:
            faces = pickle.load(f)
    
    #faces = faces[start_frame:mel_chunks_len+start_frame]
    
    return faces


def datagen(frames, mels, args):
    global start_frame_global
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    if len(frames) > 60:
        #start_frame = random.randrange(0, len(frames)-60)
        start_frame = start_frame_global
        if start_frame < 0 or start_frame >= len(frames):
            print(" -- start_frame was "+str(start_frame)+", setting it back to 0")
            start_frame = 0            
        start_frame_global = start_frame_global + len(mels)
        if start_frame_global >= len(frames):
            start_frame_global = start_frame_global - len(frames)
            print(" -- start_frame_global was >= frames, setting it back to: "+str(start_frame_global))
    else:
        start_frame = 0
        print("Warning: source video is too short for smooth chunks playing. 60+ frames minimum")
    print(str(time.time())+" Before face detection. start frame: "+str(start_frame)+", start_frame_global: "+str(start_frame_global))
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect_with_cache(frames, args, os.path.basename(args.face), os.stat(args.face).st_size) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect_with_cache([frames[0]], args, os.path.basename(args.face), os.stat(args.face).st_size)
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    print(str(time.time())+" after face detection, mels: "+str(len(mels))+", frames: "+str(len(frames))+", faces: "+str(len(face_det_results))+", start_frame: "+str(start_frame))
    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)+start_frame
        idx = idx%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch



def _load(checkpoint_path, device="cpu"):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path, device="cpu"):
    if os.path.isfile(path) == False:
        checkpoint_fname = os.path.basename(path)
        checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'checkpoints')
        if checkpoint_fname == "wav2lip.pth":
            print(f"Downloading "+checkpoint_fname+" model from huggingface into "+checkpoint_dir+"...") 
            hf_hub_download(repo_id="Ftfyhh/wav2lip", filename="wav2lip.pth", local_dir=checkpoint_dir, local_dir_use_symlinks=False)
        elif checkpoint_fname == "wav2lip_gan.pth":
            print(f"Downloading "+checkpoint_fname+" model from huggingface into "+checkpoint_dir+"...") 
            hf_hub_download(repo_id="Ftfyhh/wav2lip", filename="wav2lip_gan.pth", local_dir=checkpoint_dir, local_dir_use_symlinks=False)
        else:
            print ("Error: Download your model manually and put into /wav2lip/checkpoints/")
        
    model = Wav2Lip()
    print("Load checkpoint to "+device+" from: {}".format(path))
    checkpoint = _load(path, device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def wav2lip_main(args):
   
    print(str(time.time())+" wav2lip_main")
    #print(args)
    global device, model, full_frames_by_file
        
    device = args.device
    
    start_time = time.time()
    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True
    else:
        args.static = False
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        #fps = args.fps


        #  0.12s to read 28s video. lets store it globally in RAM for reuse
        if args.face in full_frames_by_file:
            full_frames = full_frames_by_file[args.face] # 0.001 s
        else:
            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if args.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

                #if args.rotate:
                #    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = args.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)
                full_frames_by_file[args.face] = full_frames

    if not args.audio.endswith('.wav'):
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'
    
    if (not os.path.isfile(args.audio)):
        print("Error: "+args.audio+" file is not found")
        return
    wav = audio.load_wav(args.audio, 16000)
    if (not len(wav)):
        print("out_"+str(args.chunk)+".wav len "+str(len(wav))+" is bad, exiting")
        return
        
    mel = audio.melspectrogram(wav)
    #print(str(time.time())+" after generation of "+str(args.chunk)+" mel melspectrogram "+str(mel.shape))
   

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    
    # add missing empty mels at the end
    duplicates = [mel_chunks[-1]] * 3
    mel_chunks += duplicates    
    # or at the beginning
    #duplicated_elements = [mel_chunks[0]] * 3  # Duplicate the first element thrice
    #mel_chunks = np.insert(mel_chunks, obj=0, values=duplicated_elements, axis=0)  # Insert duplicated elements at index 0
    #print(str(time.time())+" Length of mel chunks: {}".format(len(mel_chunks)))

    #full_frames = full_frames[:len(mel_chunks)]    #now we are caching faces for a full video

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks, args) # 0.00s
    #print(str(time.time())+" after datagen")

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            #print (str(time.time())+" before Model load")
            if model is None:
                #print ("Model is not loaded, loading...")
                model = load_model(args.checkpoint_path, args.device)
            print (str(time.time())+" Model loaded. Starting wav2lip inference.")

            frame_h, frame_w = full_frames[0].shape[:-1]
            # MPEG4-AVC. AVC1 may not work on all systems. try X264, AVC1, MP4V. On my system everything worked fine even without .dll, although VideoWriter was reporting errors.            
            out = cv2.VideoWriter('modules/wav2lip/temp/result_'+str(args.chunk)+'_tmp.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_w, frame_h)) # 0.00s test mp2v avc1
            #print("Download .dll of required version from https://github.com/cisco/openh264/releases and put to /system32 or /ffmpeg/bin dir. This will hide the error in console. In my case it was openh264-1.8.0-win64.dll")
            #print("cv2 created file temp/result_"+str(args.chunk)+"_tmp.mp4")
        
        #print(str(time.time())+" (mel_b:"+str(len(mel_batch))+", img_b:"+str(len(img_batch))+")")
        # last run takes 0.14s if current_batch_size is different from previous. if current_batch_size == previous: inference takes just 0.01s
        if (len(mel_batch) < batch_size):
            padding_len = batch_size - len(mel_batch)
            if padding_len:
                #print(str(time.time())+" filling last "+str(padding_len)+" mels/faces with last element to speed up batch")                
                last_img = img_batch[-1][np.newaxis, ...]  # Add an extra dimension to allow concatenation
                img_padding = np.repeat(last_img, padding_len, axis=0)
                img_batch = np.concatenate((img_batch, img_padding), axis=0)
                last_mel = mel_batch[-1][np.newaxis, ...]  # Add an extra dimension to allow concatenation
                mel_padding = np.repeat(last_mel, padding_len, axis=0)
                mel_batch = np.concatenate((mel_batch, mel_padding), axis=0)
                #print(str(time.time())+" done filling")
                
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(args.device) # 0.001 s
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(args.device) # 0.001 s
        print(str(time.time())+" before inference")
        
        # inference
        # 12 runs for a 7s audio and batch_size 16. larger batch_size needs more VRAM
        try:
            with torch.no_grad():
                pred = model(mel_batch, img_batch)  # first run 0.14 s, other runs: 0.02 s
            #print(str(time.time())+" after inference (mel_b:"+str(len(mel_batch))+", img_b:"+str(len(img_batch))+").........")
        
        except RuntimeError as err:
            #print('Runtime error:', str(err))
            #print('Warning: audio chunks in out_'+str(args.chunk)+'.wav: '+str(len(mel_chunks))+', are empty, cant predict lips => TODO copy source frames from buffer to output. Exiting.')
            out.release()
            os.replace('modules/wav2lip/temp/result_'+str(args.chunk)+'_tmp.mp4', 'modules/wav2lip/temp/result_'+str(args.chunk)+'.mp4')
            torch.cuda.empty_cache()
            return -1
            
        
        # WORKS! doing everything on a GPU, collecting images in the same frames_gpu arr. sending it to cpu only after loop.
        pred = pred * 255.0 # pred[32, 3, 96, 96]
        #print(str(time.time())+" after pred * 255")
        # Convert frames and coords to PyTorch tensors and move them to the GPU
        frames_gpu = []
        frames_gpu = [torch.from_numpy(frame).to(device) for frame in frames]   #  slow 0.13s. TODO: move all frames to GPU on app start? vram?
        #print(str(time.time())+" after moving frames to gpu")
        i = 0
        for p, f, c in zip(pred, frames_gpu, coords): # p[3, 96, 96]. 0.003 s
            y1, y2, x1, x2 = c            
            p = p.unsqueeze(dim=0)                       # Add a singleton dimension along axis 0
            if (p.shape[-1] and p.shape[-2]):
                p = F.interpolate(p, size=(int(y2 - y1), int(x2 - x1)), mode='bilinear').squeeze(dim=0).permute(1, 2, 0) # -> [96, 96, 3]                
                f[y1:y2, x1:x2] = p     # Assign the processed patch to the cloned frame
                frames_gpu[i] = f
                i+=1
                
        # After processing all frames, move them back to CPU memory. 0.003s
        #print(str(time.time())+" after GPU resizing")
        result_tensor = torch.stack(frames_gpu).cpu().numpy()
        
        # Write images to video file # 0.02 s
        for im in result_tensor:
            out.write(im)
            #print("f im "+str(len(im)))
        #print(str(time.time())+" after writing temp/result_"+str(args.chunk)) 
        
        '''
        # old, original, slower. WORKING        
        # GPU tensor to CPU array
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255. # 0.07s. check https://docs.cupy.dev/en/stable/ if it is faster then numpy
        # resize, replace, save
        print(str(time.time())+" after cpu().transpose")
        for p, f, c in zip(pred, frames, coords): # 0.01s (total)
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            print( "p size: "+str(len(p)))
            f[y1:y2, x1:x2] = p #replaces the original subregion with prediction (p).
            out.write(f)
        print(str(time.time())+" after replacing and writing")
        '''
        
    out.release()
    os.replace('modules/wav2lip/temp/result_'+str(args.chunk)+'_tmp.mp4', 'modules/wav2lip/temp/result_'+str(args.chunk)+'.mp4')
    #print(str(time.time())+" after inference of mixing mels and faces, before playing")
        
    #print("video is saved. This thread is not for playback, exiting.")
    #command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'modules/wav2lip/temp/result.avi', args.outfile) # mp4:0.30-0.23s.  avi: 0.12-0.13s
    #command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 -map 1:v -map 0:a -codec copy {}'.format(args.audio, 'modules/wav2lip/temp/result_'+str(args.chunk)+'.ts', args.outfile) # mp4 copy: 0.04s, plays in chrome on win11, todo: test in android
    #command = 'ffmpeg -y -i {} -i {} -g 60 -hls_time 2 -hls_list_size 0 -start_number {} {}'.format(args.audio, 'modules/wav2lip/temp/result_'+str(args.chunk)+'.mp4', args.chunk, args.outfile) # mp4 copy: 0.04s, plays in chrome on win11, todo: test in android
    
    #subprocess.call(command, shell=platform.system() != 'Windows')
    torch.cuda.empty_cache()
    print (str(time.time())+" wav2lip done with "+args.outfile+" in "+str(round(time.time()-start_time, 2))+" s.")
    print (" ")
    print (" ")
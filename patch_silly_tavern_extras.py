import os, time, shutil, codecs


def return_patch_files():
    file_paths_with_replacements = [
        ['../../server.py', 
            {'app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024': """app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


if "wav2lip" in modules:
    sys.path.append("modules/wav2lip/")
    from server_wav2lip import *
        
    @app.route("/api/wav2lip/generate", methods=["GET","POST"]) 
    @app.route("/api/wav2lip/generate/<char_folder>", methods=["GET","POST"]) 
    @app.route("/api/wav2lip/generate/<char_folder>/<device>", methods=["GET","POST"]) 
    @app.route("/api/wav2lip/generate/<char_folder>/<device>/<audio>", methods=["GET","POST"]) 
    @cross_origin(headers=['Content-Type'])
    def wav2lip_generate(char_folder="default",device="cpu",audio="test"):
        return wav2lip_server_generate(char_folder, device, audio)
    
    @app.route("/api/wav2lip/play/<char_folder>/<fname>", methods=["GET","POST"]) 
    @cross_origin(headers=['Content-Type'])
    def wav2lip_play(fname: str, char_folder: str):
        return wav2lip_server_play(fname, char_folder)
        
    @app.route("/api/wav2lip/silero_set_lang/<fname>", methods=["GET","POST"]) 
    @cross_origin(headers=['Content-Type'])
    def wav2lip_silero_set_lang(fname: str):
        return wav2lip_server_silero_set_lang(tts_service, fname)
    
    @app.route("/api/wav2lip/get_chars", methods=["GET","POST"]) 
    @cross_origin(headers=['Content-Type'])
    def wav2lip_get_chars():
        return wav2lip_server_get_chars()         
    
    # override old generate
    @app.route("/api/tts/generate", methods=["POST"])
    def wav2lip_tts_generate():
        voice = request.get_json()
        return wav2lip_server_tts_generate(tts_service, voice)
    """,
    
            'streaming_module.whisper_model, streaming_module.vosk_model = streaming_module.load_model(file_path=whisper_model_path)': '''#streaming_module.whisper_model, streaming_module.vosk_model = streaming_module.load_model(file_path=whisper_model_path)'''}
        ],
    ]
    
    return file_paths_with_replacements
    

def patch_files(file_paths_with_replacements):
    for file_path_with_replacements in file_paths_with_replacements:
        file_path = file_path_with_replacements[0]
        if os.path.exists(file_path):
            replacements = file_path_with_replacements[1]
            
            # Check if file has already been patched
            backup_file_path = file_path + '.bkp'
            if os.path.exists(backup_file_path):
                print(file_path+" was already patched before (.bkp file exists. If you want to patch it again - first restore the original file), skipping.")
                continue
            
            # Create backup copy of original file
            shutil.copy(file_path, backup_file_path)
            
            # Make replacements in file
            with codecs.open(file_path, encoding='utf-8', mode='r+') as f:
                file_contents = f.read()
                
                for needle, replacement in replacements.items():
                    file_contents = file_contents.replace(needle, replacement)
                    
                f.seek(0)
                f.write(file_contents)
                f.close()
                print(file_path+" successfully patched.")
        else:
            print(file_path+" is not found, skipping.")

    
if __name__ == "__main__":
    
    
    patch_files(return_patch_files())
    print ("\n\nSuccess. closing this window in 60 seconds.")
    time.sleep(60)
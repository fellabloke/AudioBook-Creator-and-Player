# download_model.py
import os
from TTS.utils.manage import ModelManager
import nltk

# --- Coqui TTS Model Download (Your existing logic) ---
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2" 
print(f"Attempting to download/verify Coqui TTS model: {MODEL_NAME}")
try:
    manager = ModelManager()
    model_path, config_path, model_item = manager.download_model(MODEL_NAME)
    print(f"--- Coqui TTS Download Script Paths ---") # For debugging
    print(f"Model Path: {model_path}")
    print(f"Config Path: {config_path}")
    if not (model_path and os.path.exists(model_path)):
        print(f"CRITICAL: Coqui Model file NOT found at {model_path} after download attempt.")
        # (Optional: list available models from manager if you want to debug model names)
        import sys
        sys.exit(1)
    print(f"Coqui model {MODEL_NAME} confirmed/downloaded.")
except Exception as e:
    print(f"CRITICAL Error during Coqui TTS model download for {MODEL_NAME}: {e}")
    import sys
    sys.exit(1)
# --- End Coqui TTS Model Download ---

# --- NLTK Data Download ---
# NLTK_DATA_PACKAGES = ['punkt'] # Original
NLTK_DATA_PACKAGES = ['punkt', 'punkt_tab'] # <<< TRY ADDING punkt_tab
print(f"Attempting to download NLTK data packages: {NLTK_DATA_PACKAGES}")
try:
    for pkg in NLTK_DATA_PACKAGES:
        print(f"Downloading NLTK package: {pkg}")
        nltk.download(pkg, quiet=False) # quiet=False to see output
    print(f"NLTK data packages {NLTK_DATA_PACKAGES} downloaded successfully or already present.")
except Exception as e:
    print(f"Error downloading NLTK data packages: {e}")
    # Fail build if 'punkt' (the primary one) is missing
    if 'punkt' in NLTK_DATA_PACKAGES and not os.path.exists(os.path.join(nltk.downloader.Downloader().default_download_dir(), 'tokenizers', 'punkt')):
        print("CRITICAL: Main 'punkt' tokenizer data failed to download. Sentence tokenization will likely fail.")
        import sys
        sys.exit(1)
    # If 'punkt_tab' specifically failed, we can log it but maybe not fail the build if 'punkt' is there.
    # However, since the error asked for it, let's treat its absence as critical for now.
    if 'punkt_tab' in NLTK_DATA_PACKAGES and not os.path.exists(os.path.join(nltk.downloader.Downloader().default_download_dir(), 'tokenizers', 'punkt_tab')): # Path might be different
        print("CRITICAL: 'punkt_tab' resource failed to download as requested by error message.")
        import sys
        sys.exit(1)

print("All model and data download scripts finished successfully.")
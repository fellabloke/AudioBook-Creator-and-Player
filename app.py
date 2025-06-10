import os
import uuid
import fitz  # PyMuPDF
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import nltk
import torch
import threading
import logging
import re
import time
import unicodedata
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([
        XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs
    ])
    logger.info("Successfully added required TTS classes to torch safe globals.")
except (ImportError, AttributeError) as e_safe_globals:
    logger.warning(f"Could not configure PyTorch safe globals: {e_safe_globals}. Model loading might fail.")

from TTS.api import TTS

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads_temp'
SPEAKER_SAMPLES_FOLDER = os.path.join(UPLOAD_FOLDER, 'speaker_samples')
CHUNK_AUDIO_FOLDER = 'static/chunk_audio'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPEAKER_SAMPLES_FOLDER'] = SPEAKER_SAMPLES_FOLDER
app.config['CHUNK_AUDIO_FOLDER'] = CHUNK_AUDIO_FOLDER
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024 # 256 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPEAKER_SAMPLES_FOLDER, exist_ok=True)
os.makedirs(CHUNK_AUDIO_FOLDER, exist_ok=True)

SESSIONS = {}
session_modification_lock = threading.Lock()

TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_ENABLED = False
tts_instance = None
USE_GPU = torch.cuda.is_available()

logger.info(f"Attempting to load Coqui TTS model: {TTS_MODEL_NAME}")
if USE_GPU: logger.info("CUDA is available! Attempting to load TTS model on GPU.")
else: logger.info("CUDA not available. Loading TTS model on CPU. Check Docker GPU setup if GPU is expected.")

try:
    tts_instance = TTS(model_name=TTS_MODEL_NAME, progress_bar=True, gpu=USE_GPU)
    logger.info(f"Coqui TTS model '{TTS_MODEL_NAME}' loaded successfully (on {'GPU' if USE_GPU else 'CPU'}).")
    TTS_ENABLED = True
except Exception as e:
    logger.critical(f"Error loading Coqui TTS model '{TTS_MODEL_NAME}': {e}", exc_info=True)
    if USE_GPU and "CUDA" in str(e).upper():
        logger.warning("GPU loading failed. Ensure Docker has GPU access & PyTorch CUDA version is correct.")

MIN_TEXT_LEN_FOR_TTS = 10
APPROX_OPTIMAL_TTS_CHARS = 250
MAX_TTS_CHUNK_LEN = 280
MAX_CHUNK_SYNTHESIS_ATTEMPTS = 3

def clean_text_for_tts(text_input):
    if not text_input or not isinstance(text_input, str): return ""
    text = text_input
    specific_replacements = {
        '\u00A0': ' ', '\u00AD': '', '\u200B': '', '\u2018': "'", '\u2019': "'",
        '\u201C': '"', '\u201D': '"', '\u2013': '-', '\u2014': '-'
    }
    for char, replacement in specific_replacements.items(): text = text.replace(char, replacement)
    text = unicodedata.normalize('NFC', text)
    cleaned_chars = []
    for char_val in text:
        cat = unicodedata.category(char_val)
        if cat.startswith('C') and char_val not in ['\n', '\r', '\t']: continue
        cleaned_chars.append(char_val)
    text = "".join(cleaned_chars)
    text = re.sub(r'\s\s+', ' ', text).strip()
    text = re.sub(r'(\r\n|\r|\n){2,}', '\n\n', text)
    text = text.replace('\n', ' ')
    if 0 < len(text) < 20 and not re.search(r'[a-zA-Z0-9]{3,}', text):
        if not any(c.isalnum() for c in text): return ""
    return text.strip()

def extract_and_chunk_pdf_with_page_info(pdf_filepath, chunk_by='paragraph'):
    processed_chunks = []
    try:
        doc = fitz.open(pdf_filepath)
        full_doc_text_for_sentence_tokenization = ""
        page_texts_for_paragraph_chunking = []
        if chunk_by == 'paragraph':
            for page_num_zero_based, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text.strip():
                    page_texts_for_paragraph_chunking.append({'text': page_text, 'page_num': page_num_zero_based + 1})
        else:
            for page in doc:
                full_doc_text_for_sentence_tokenization += page.get_text("text") + "\n "
        doc.close()

        if chunk_by == 'paragraph':
            for page_data in page_texts_for_paragraph_chunking:
                cleaned_page_text = clean_text_for_tts(page_data['text'])
                page_paragraphs = [p.strip() for p in cleaned_page_text.split('\n\n') if p.strip()]
                for p_text in page_paragraphs:
                    if p_text: processed_chunks.append({'text': p_text, 'page': page_data['page_num']})
            if not processed_chunks and any(pt['text'].strip() for pt in page_texts_for_paragraph_chunking):
                logger.warning("No paragraph chunks found from '\n\n' splits, creating one chunk per page from cleaned page text.")
                for pt_data in page_texts_for_paragraph_chunking:
                    cleaned_text_for_page = clean_text_for_tts(pt_data['text']).strip()
                    if cleaned_text_for_page: processed_chunks.append({'text': cleaned_text_for_page, 'page': pt_data['page_num']})
        elif chunk_by == 'sentence':
            cleaned_full_doc_text = clean_text_for_tts(full_doc_text_for_sentence_tokenization)
            if not cleaned_full_doc_text.strip(): return []
            sentences = nltk.sent_tokenize(cleaned_full_doc_text)
            processed_chunks = [{'text': s.strip(), 'page': -1} for s in sentences if s.strip()]
            if not processed_chunks and cleaned_full_doc_text.strip():
                processed_chunks = [{'text': cleaned_full_doc_text.strip(), 'page': -1}]

        if not processed_chunks:
            try:
                doc_raw = fitz.open(pdf_filepath)
                raw_full_text = "".join([page.get_text("text") for page in doc_raw])
                doc_raw.close()
                cleaned_raw_full_text = clean_text_for_tts(raw_full_text).strip()
                if cleaned_raw_full_text:
                    logger.warning("No chunks by chosen method; using cleaned PDF text as one chunk.")
                    processed_chunks = [{'text': cleaned_raw_full_text, 'page': 1}]
            except Exception as e_raw_extract:
                logger.error(f"Could not extract raw text from PDF {pdf_filepath} as a last resort: {e_raw_extract}")
                return []
        return processed_chunks
    except Exception as e:
        logger.error(f"Error in extract_and_chunk_pdf_with_page_info for '{pdf_filepath}': {e}", exc_info=True)
        return []

def identify_skippable_chunks_and_main_content(chunk_data_list):
    preliminary_keywords_lower = [
        "contents", "table of contents", "acknowledgments", "acknowledgements",
        "dedication", "preface", "introduction", "abstract", "summary",
        "glossary", "index", "bibliography", "references"
    ]
    main_content_indicators_lower = ["chapter", "part ", "section ", "unit "]
    skippable_indices = []
    main_content_start_index = 0
    found_main_content_flag = False

    for i, chunk_item in enumerate(chunk_data_list):
        chunk_text_lower_lines = chunk_item['text'].lower().splitlines()
        first_few_lines_lower = "\n".join(chunk_text_lower_lines[:3])
        is_prelim_keyword = any(keyword in first_few_lines_lower for keyword in preliminary_keywords_lower)
        is_likely_toc = False
        if not is_prelim_keyword and 3 < len(chunk_text_lower_lines) < 60 :
            num_lines_ending_page_like = 0
            for line_idx, line in enumerate(chunk_text_lower_lines):
                stripped_line = line.strip()
                if not stripped_line: continue
                if stripped_line and stripped_line[-1].isdigit():
                    potential_page_num_part = stripped_line.split()[-1]
                    if re.fullmatch(r'([ivxlcdm]+|[0-9]+([\.\-][0-9ivxlcdm]+)*)', potential_page_num_part.replace('.', '')):
                        num_lines_ending_page_like +=1
            if num_lines_ending_page_like > len(chunk_text_lower_lines) * 0.35:
                is_likely_toc = True

        if is_prelim_keyword or is_likely_toc:
            skippable_indices.append(i)
            if not found_main_content_flag:
                main_content_start_index = i + 1
        elif any(indicator in first_few_lines_lower for indicator in main_content_indicators_lower):
            if not found_main_content_flag:
                 main_content_start_index = i
                 found_main_content_flag = True
        elif not found_main_content_flag:
            main_content_start_index = i
            found_main_content_flag = True

    while main_content_start_index in skippable_indices and main_content_start_index < len(chunk_data_list) -1 :
        main_content_start_index += 1
    if main_content_start_index >= len(chunk_data_list) and chunk_data_list:
        main_content_start_index = 0
    return skippable_indices, main_content_start_index

@app.route('/', methods=['GET'])
def index(): return render_template('index.html', tts_enabled=TTS_ENABLED)

@app.route('/player')
def player_page():
    session_id = request.args.get('session_id')
    with session_modification_lock:
        if not session_id or session_id not in SESSIONS:
            logger.warning(f"Player page requested for invalid/expired session: {session_id}")
            return redirect(url_for('index'))
    return render_template('player.html', session_id=session_id)

@app.route('/initiate_processing_session', methods=['POST'])
def initiate_processing_session():
    if not TTS_ENABLED: return jsonify({"error": "TTS service not available."}), 503
    if 'pdf_file' not in request.files or 'voice_sample' not in request.files:
        return jsonify({"error": "PDF and voice sample required."}), 400

    pdf_file = request.files['pdf_file']; voice_sample_file = request.files['voice_sample']
    if pdf_file.filename == '' or voice_sample_file.filename == '':
        return jsonify({"error": "Files not selected."}), 400
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid PDF type. Please upload a .pdf file."}), 400
    if not voice_sample_file.filename.lower().endswith(('.wav', '.mp3')):
        return jsonify({"error": "Invalid voice sample format. Please use .wav or .mp3."}), 400

    pdf_fn_secure = secure_filename(pdf_file.filename)
    temp_pdf_fn = f"temp_{uuid.uuid4().hex[:8]}_{pdf_fn_secure}"
    temp_pdf_fp = os.path.join(app.config['UPLOAD_FOLDER'], temp_pdf_fn)
    pdf_file.save(temp_pdf_fp)

    speaker_fn_secure = secure_filename(voice_sample_file.filename)
    speaker_fn = f"speaker_{uuid.uuid4().hex[:8]}{os.path.splitext(speaker_fn_secure)[1]}"
    speaker_fp = os.path.join(app.config['SPEAKER_SAMPLES_FOLDER'], speaker_fn)
    voice_sample_file.save(speaker_fp)

    processed_chunks_data_with_page_info = extract_and_chunk_pdf_with_page_info(temp_pdf_fp, chunk_by='paragraph')

    if os.path.exists(temp_pdf_fp):
        try: os.remove(temp_pdf_fp)
        except Exception as e_rem_pdf: logger.warning(f"Could not remove temp PDF {temp_pdf_fp}: {e_rem_pdf}")

    if not processed_chunks_data_with_page_info:
        if os.path.exists(speaker_fp):
            try: os.remove(speaker_fp)
            except Exception as e_rem_spk: logger.warning(f"Could not remove speaker sample {speaker_fp} after PDF failure: {e_rem_spk}")
        return jsonify({"error": "Could not extract any text chunks from PDF. The PDF might be image-based, empty, or text became empty after cleaning."}), 500

    text_chunks_for_synthesis = [item['text'] for item in processed_chunks_data_with_page_info]
    skippable_indices, main_content_start_idx = identify_skippable_chunks_and_main_content(processed_chunks_data_with_page_info)

    session_id = str(uuid.uuid4())
    with session_modification_lock:
        SESSIONS[session_id] = {
            "pdf_filename_original": pdf_fn_secure,
            "speaker_wav_filename": speaker_fn,
            "text_chunks": text_chunks_for_synthesis,
            "chunk_audio_filenames": [None] * len(text_chunks_for_synthesis),
            "synthesis_status": ["pending"] * len(text_chunks_for_synthesis),
            "synthesis_retry_count": [0] * len(text_chunks_for_synthesis),
            "skippable_indices": skippable_indices,
            "page_info_per_chunk": [item.get('page', -1) for item in processed_chunks_data_with_page_info]
        }
    logger.info(f"Session {session_id} initiated for '{pdf_fn_secure}' with {len(text_chunks_for_synthesis)} chunks. Main content estimated at index {main_content_start_idx}.")

    return jsonify({
        "success": True,
        "session_id": session_id,
        "num_chunks": len(text_chunks_for_synthesis),
        "original_filename": pdf_fn_secure,
        "main_content_start_index": main_content_start_idx
    })

@app.route('/get_chunk_metadata/<session_id>', methods=['GET'])
def get_chunk_metadata(session_id):
    with session_modification_lock:
        session_data = SESSIONS.get(session_id)
        if not session_data:
            return jsonify({"error": "Session not found"}), 404

        num_chunks = len(session_data.get("text_chunks", []))
        skippable_indices = session_data.get("skippable_indices", [])
        page_info_per_chunk = session_data.get("page_info_per_chunk", [])
        
        chunk_previews_for_ui = []
        for i in range(num_chunks):
            text_for_snippet = session_data["text_chunks"][i]
            chunk_previews_for_ui.append({
                "id": i,
                "text_snippet": text_for_snippet[:100] + "..." if len(text_for_snippet) > 100 else text_for_snippet,
                "page": page_info_per_chunk[i] if i < len(page_info_per_chunk) else -1,
                "is_skippable": i in skippable_indices
            })
        
        return jsonify({
            "success": True,
            "chunk_previews": chunk_previews_for_ui
        })

def _do_synthesize_chunk_with_subsplitting(session_id, chunk_index_to_synth, is_background_thread=False):
    thread_prefix = "[BG_SubSplit]" if is_background_thread else "[FG_SubSplit]"
    logger.info(f"{thread_prefix} START synthesis for S:{session_id} C:{chunk_index_to_synth}")
    original_chunk_text = ""
    speaker_wav_filename = ""

    with session_modification_lock:
        session_data = SESSIONS.get(session_id)
        if not session_data:
            logger.error(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - Session not found during synth.")
            return False, "Session not found"
        if not 0 <= chunk_index_to_synth < len(session_data["text_chunks"]):
            logger.error(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - Invalid chunk index.")
            return False, "Invalid chunk index"
        original_chunk_text = session_data["text_chunks"][chunk_index_to_synth]
        speaker_wav_filename = session_data["speaker_wav_filename"]

    text_for_this_chunk = original_chunk_text

    if not text_for_this_chunk.strip() or len(text_for_this_chunk.strip()) < MIN_TEXT_LEN_FOR_TTS:
        error_msg = f"Chunk text empty/too short (min {MIN_TEXT_LEN_FOR_TTS}). Actual: {len(text_for_this_chunk.strip())}. Text: '{text_for_this_chunk[:70]}...'"
        logger.warning(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - {error_msg}")
        with session_modification_lock:
            session_data = SESSIONS.get(session_id) # Re-get for update
            if session_data and 0 <= chunk_index_to_synth < len(session_data["synthesis_status"]):
                session_data["synthesis_status"][chunk_index_to_synth] = "error"
                session_data["chunk_audio_filenames"][chunk_index_to_synth] = "empty_or_too_short_text_error"
        logger.info(f"{thread_prefix} END synthesis for S:{session_id} C:{chunk_index_to_synth} with error: {error_msg}")
        return False, error_msg

    sub_texts_to_synthesize = []
    if len(text_for_this_chunk) > APPROX_OPTIMAL_TTS_CHARS:
        logger.info(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} (len {len(text_for_this_chunk)}) > optimal {APPROX_OPTIMAL_TTS_CHARS}, sub-splitting.")
        sentences = nltk.sent_tokenize(text_for_this_chunk)
        current_sub_chunk_text = ""
        for sentence_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence: continue
            if len(current_sub_chunk_text) + len(sentence) + 1 <= APPROX_OPTIMAL_TTS_CHARS:
                current_sub_chunk_text += (sentence + " ")
            else:
                if current_sub_chunk_text.strip():
                    sub_texts_to_synthesize.append(current_sub_chunk_text.strip())
                if len(sentence) <= APPROX_OPTIMAL_TTS_CHARS:
                    current_sub_chunk_text = sentence + " "
                else:
                    logger.warning(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} Sent:{sentence_idx} (len {len(sentence)}) > optimal {APPROX_OPTIMAL_TTS_CHARS}. Hard splitting.")
                    start = 0
                    while start < len(sentence):
                        sub_texts_to_synthesize.append(sentence[start : start + MAX_TTS_CHUNK_LEN])
                        start += MAX_TTS_CHUNK_LEN
                    current_sub_chunk_text = ""
        if current_sub_chunk_text.strip(): sub_texts_to_synthesize.append(current_sub_chunk_text.strip())
        if not sub_texts_to_synthesize:
            logger.warning(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - NLTK split yielded no sub-chunks. Fallback to hard split.")
            start = 0
            while start < len(text_for_this_chunk):
                sub_texts_to_synthesize.append(text_for_this_chunk[start : start + MAX_TTS_CHUNK_LEN])
                start += MAX_TTS_CHUNK_LEN
    else:
        sub_texts_to_synthesize.append(text_for_this_chunk)

    if not sub_texts_to_synthesize:
        err_no_sub = "No text to synthesize after all processing."
        logger.error(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - {err_no_sub}")
        with session_modification_lock:
            session_data = SESSIONS.get(session_id)
            if session_data and 0 <= chunk_index_to_synth < len(session_data["synthesis_status"]):
                 session_data["synthesis_status"][chunk_index_to_synth] = "error"
                 session_data["chunk_audio_filenames"][chunk_index_to_synth] = "no_text_after_subsplitting"
        logger.info(f"{thread_prefix} END synthesis for S:{session_id} C:{chunk_index_to_synth} with error: {err_no_sub}")
        return False, err_no_sub

    sub_chunk_audio_temp_files = []
    all_sub_chunks_successful = True
    last_sub_chunk_error_detail = "Unknown sub-chunk synthesis error"
    logger.info(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - Will synthesize {len(sub_texts_to_synthesize)} sub-chunk(s).")

    for i, sub_text_to_synth_raw in enumerate(sub_texts_to_synthesize):
        sub_text_to_synth = sub_text_to_synth_raw
        if len(sub_text_to_synth) < MIN_TEXT_LEN_FOR_TTS:
            logger.info(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} Sub:{i+1} (len {len(sub_text_to_synth)}) too short, skipping.")
            continue
        if len(sub_text_to_synth) > MAX_TTS_CHUNK_LEN:
            logger.warning(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} Sub:{i+1} (len {len(sub_text_to_synth)}) > max {MAX_TTS_CHUNK_LEN}, truncating (safeguard).")
            sub_text_to_synth = sub_text_to_synth[:MAX_TTS_CHUNK_LEN] + " (final truncate)"

        # logger.info(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} Sub:{i+1} - Synthesizing. Len: {len(sub_text_to_synth)}. Text: '{sub_text_to_synth[:30]}...'")
        sub_chunk_audio_basefn = f"{session_id}_chunk_{chunk_index_to_synth}_sub_{i}_temp.wav"
        sub_chunk_audio_fp = os.path.join(app.config['CHUNK_AUDIO_FOLDER'], sub_chunk_audio_basefn)
        speaker_wav_full_fp = os.path.join(app.config['SPEAKER_SAMPLES_FOLDER'], speaker_wav_filename)

        sub_synthesis_success_current = False
        for attempt in range(MAX_CHUNK_SYNTHESIS_ATTEMPTS):
            logger.info(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} Sub:{i+1} Attempt {attempt+1}...")
            try:
                if not tts_instance or not TTS_ENABLED: raise Exception("TTS instance not available.")
                if not os.path.exists(speaker_wav_full_fp): raise FileNotFoundError(f"Speaker WAV missing: {speaker_wav_full_fp}")
                
                # Log the exact text being sent to TTS
                logger.debug(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} Sub:{i+1} Attempt {attempt+1} TTS_INPUT: '{sub_text_to_synth}'")

                tts_instance.tts_to_file(text=sub_text_to_synth, speaker_wav=speaker_wav_full_fp, language="en", file_path=sub_chunk_audio_fp)
                sub_chunk_audio_temp_files.append(sub_chunk_audio_fp)
                sub_synthesis_success_current = True
                logger.info(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} Sub:{i+1} Attempt {attempt+1} SUCCEEDED.")
                break
            except Exception as e_sub_synth:
                last_sub_chunk_error_detail = str(e_sub_synth)
                logger.error(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} Sub:{i+1} Attempt {attempt+1} FAILED: {e_sub_synth}", exc_info= (attempt == MAX_CHUNK_SYNTHESIS_ATTEMPTS - 1) )
                if os.path.exists(sub_chunk_audio_fp):
                    try: os.remove(sub_chunk_audio_fp)
                    except Exception as e_rm_sub: logger.error(f"{thread_prefix} Error removing failed sub-audio {sub_chunk_audio_fp}: {e_rm_sub}")
                if attempt < MAX_CHUNK_SYNTHESIS_ATTEMPTS - 1: time.sleep(0.5 + attempt * 0.5)
                else: all_sub_chunks_successful = False
        if not sub_synthesis_success_current: all_sub_chunks_successful = False; break

    if not all_sub_chunks_successful or not sub_chunk_audio_temp_files:
        err_sub_synth_fail = f"Not all sub-chunks synthesized or no valid sub-chunks. Last error: {last_sub_chunk_error_detail}"
        logger.error(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - {err_sub_synth_fail}")
        for fp_to_clean in sub_chunk_audio_temp_files:
            if os.path.exists(fp_to_clean):
                try: os.remove(fp_to_clean)
                except Exception as e_clean: logger.error(f"Error cleaning temp sub-audio {fp_to_clean}: {e_clean}")
        with session_modification_lock:
            session_data = SESSIONS.get(session_id)
            if session_data and 0 <= chunk_index_to_synth < len(session_data["synthesis_status"]):
                session_data["synthesis_status"][chunk_index_to_synth] = "error"
                session_data["chunk_audio_filenames"][chunk_index_to_synth] = f"sub_chunk_synth_failed: {last_sub_chunk_error_detail[:200]}"
        logger.info(f"{thread_prefix} END synthesis for S:{session_id} C:{chunk_index_to_synth} with error: {err_sub_synth_fail}")
        return False, f"Sub-chunk synthesis failed: {last_sub_chunk_error_detail[:200]}"

    final_chunk_audio_basefn = f"{session_id}_chunk_{chunk_index_to_synth}.wav"
    final_chunk_audio_fp = os.path.join(app.config['CHUNK_AUDIO_FOLDER'], final_chunk_audio_basefn)

    if len(sub_chunk_audio_temp_files) == 1:
        logger.info(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - Single sub-chunk, renaming {sub_chunk_audio_temp_files[0]} to {final_chunk_audio_fp}")
        try:
            os.rename(sub_chunk_audio_temp_files[0], final_chunk_audio_fp)
        except Exception as e_rename:
            err_rename = f"Audio rename failed: {e_rename}"
            logger.error(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - {err_rename}")
            if os.path.exists(sub_chunk_audio_temp_files[0]): os.remove(sub_chunk_audio_temp_files[0])
            if os.path.exists(final_chunk_audio_fp): os.remove(final_chunk_audio_fp)
            with session_modification_lock:
                session_data = SESSIONS.get(session_id)
                if session_data and 0 <= chunk_index_to_synth < len(session_data["synthesis_status"]):
                    session_data["synthesis_status"][chunk_index_to_synth] = "error"
                    session_data["chunk_audio_filenames"][chunk_index_to_synth] = f"audio_rename_failed: {str(e_rename)[:150]}"
            logger.info(f"{thread_prefix} END synthesis for S:{session_id} C:{chunk_index_to_synth} with error: {err_rename}")
            return False, f"Audio rename failed: {e_rename}"
    else:
        logger.info(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - Concatenating {len(sub_chunk_audio_temp_files)} sub-chunks.")
        concat_list_fn = f"concat_list_{session_id}_{chunk_index_to_synth}.txt"
        concat_list_fp = os.path.join(app.config['UPLOAD_FOLDER'], concat_list_fn)
        try:
            with open(concat_list_fp, 'w', encoding='utf-8') as f:
                for sub_fp in sub_chunk_audio_temp_files: f.write(f"file '{os.path.abspath(sub_fp)}'\n")
            ffmpeg_command = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_fp, '-c', 'copy', final_chunk_audio_fp]
            logger.info(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - FFMPEG command: {' '.join(ffmpeg_command)}")
            process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=False)
            if process.returncode != 0:
                logger.error(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - FFMPEG FAILED. RC: {process.returncode}. STDOUT: {process.stdout}. STDERR: {process.stderr}")
                raise subprocess.CalledProcessError(process.returncode, ffmpeg_command, output=process.stdout, stderr=process.stderr)
            logger.info(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - FFMPEG successful.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e_concat:
            err_ffmpeg = f"FFMPEG concat failed: {e_concat.stderr[:300] if hasattr(e_concat, 'stderr') else str(e_concat)}"
            logger.error(f"{thread_prefix} S:{session_id} C:{chunk_index_to_synth} - {err_ffmpeg}")
            if os.path.exists(final_chunk_audio_fp): os.remove(final_chunk_audio_fp)
            with session_modification_lock:
                session_data = SESSIONS.get(session_id)
                if session_data and 0 <= chunk_index_to_synth < len(session_data["synthesis_status"]):
                    session_data["synthesis_status"][chunk_index_to_synth] = "error"
                    session_data["chunk_audio_filenames"][chunk_index_to_synth] = f"ffmpeg_concat_failed: {str(e_concat)[:150]}"
            logger.info(f"{thread_prefix} END synthesis for S:{session_id} C:{chunk_index_to_synth} with error: {err_ffmpeg}")
            return False, f"FFMPEG concatenation failed: {str(e_concat)[:150]}"
        finally:
            if os.path.exists(concat_list_fp): os.remove(concat_list_fp)
            for sub_fp in sub_chunk_audio_temp_files:
                if os.path.exists(sub_fp):
                    try: os.remove(sub_fp)
                    except Exception as e_rm_final_sub: logger.warning(f"Error removing temp sub-audio {sub_fp}: {e_rm_final_sub}")

    with session_modification_lock:
        session_data = SESSIONS.get(session_id)
        if session_data and 0 <= chunk_index_to_synth < len(session_data["synthesis_status"]):
            session_data["chunk_audio_filenames"][chunk_index_to_synth] = final_chunk_audio_basefn
            session_data["synthesis_status"][chunk_index_to_synth] = "done"
            session_data["synthesis_retry_count"][chunk_index_to_synth] = 0
    logger.info(f"{thread_prefix} END synthesis for S:{session_id} C:{chunk_index_to_synth} SUCCESSFULLY. Final audio: {final_chunk_audio_basefn}")
    return True, None

def synthesize_chunk_in_background(session_id, chunk_index_to_synth):
    bg_lock_flag_in_session = f"_bg_synthesizing_chunk_{chunk_index_to_synth}_"
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] BG THREAD STARTS for S:{session_id} C:{chunk_index_to_synth}")

    synthesis_initiated_by_this_thread = False
    try:
        with session_modification_lock:
            session_data = SESSIONS.get(session_id)
            if not session_data:
                logger.info(f"[{thread_name}] S:{session_id} C:{chunk_index_to_synth} - Session gone before BG synth lock acquisition.")
                return # Exit thread

            if not 0 <= chunk_index_to_synth < len(session_data.get("text_chunks", [])):
                logger.warning(f"[{thread_name}] S:{session_id} C:{chunk_index_to_synth} - Invalid chunk index for BG synth.")
                session_data.pop(bg_lock_flag_in_session, None) # Clean up if somehow set by a previous failed attempt
                return # Exit thread

            if session_data["synthesis_status"][chunk_index_to_synth] != "pending":
                logger.info(f"[{thread_name}] S:{session_id} C:{chunk_index_to_synth} - Not pending (status: {session_data['synthesis_status'][chunk_index_to_synth]}). BG synth not proceeding.")
                # If it was locked by this thread but status changed by another, clear lock
                if session_data.get(bg_lock_flag_in_session, False):
                     logger.warning(f"[{thread_name}] S:{session_id} C:{chunk_index_to_synth} - BG lock was set, but status is now {session_data['synthesis_status'][chunk_index_to_synth]}. Clearing lock.")
                     session_data.pop(bg_lock_flag_in_session, None)
                return # Exit thread

            if session_data.get(bg_lock_flag_in_session, False): # Should ideally not happen if called correctly
                logger.info(f"[{thread_name}] S:{session_id} C:{chunk_index_to_synth} - BG synth lock ALREADY SET. Another BG thread likely active. Exiting.")
                return # Exit thread
            
            # If we reach here, chunk is 'pending' and not locked by a BG task.
            session_data[bg_lock_flag_in_session] = True # Acquire lock
            session_data["synthesis_status"][chunk_index_to_synth] = "synthesizing"
            synthesis_initiated_by_this_thread = True
            logger.info(f"[{thread_name}] S:{session_id} C:{chunk_index_to_synth} - BG synth lock ACQUIRED. Status set to 'synthesizing'.")

        # Perform synthesis outside the initial lock if acquired
        if synthesis_initiated_by_this_thread:
            _do_synthesize_chunk_with_subsplitting(session_id, chunk_index_to_synth, is_background_thread=True)

    except Exception as e_bg_wrapper: 
        logger.error(f"[{thread_name}] S:{session_id} C:{chunk_index_to_synth} - UNHANDLED EXCEPTION in BG synth wrapper: {e_bg_wrapper}", exc_info=True)
        # Ensure status is error on such unexpected failure
        with session_modification_lock:
            session_data = SESSIONS.get(session_id)
            if session_data and 0 <= chunk_index_to_synth < len(session_data["synthesis_status"]):
                # Only update if it's still 'synthesizing' (i.e., _do_synthesize didn't already set it to error)
                if session_data["synthesis_status"][chunk_index_to_synth] == "synthesizing":
                    session_data["synthesis_status"][chunk_index_to_synth] = "error"
                    session_data["chunk_audio_filenames"][chunk_index_to_synth] = f"bg_thread_wrapper_error: {str(e_bg_wrapper)[:100]}"
                    logger.error(f"[{thread_name}] S:{session_id} C:{chunk_index_to_synth} - Status set to 'error' due to wrapper exception.")
    finally:
        # This finally block ensures the lock is released if this thread was the one that set it.
        if synthesis_initiated_by_this_thread:
            with session_modification_lock:
                session_data = SESSIONS.get(session_id)
                if session_data:
                    popped_flag = session_data.pop(bg_lock_flag_in_session, None)
                    logger.info(f"[{thread_name}] S:{session_id} C:{chunk_index_to_synth} - BG synth lock released in finally. Popped: {popped_flag is not None}")
        logger.info(f"[{thread_name}] BG THREAD ENDS for S:{session_id} C:{chunk_index_to_synth}")


@app.route('/get_audio_chunk/<session_id>/<int:chunk_index>', methods=['GET'])
def get_audio_chunk(session_id, chunk_index):
    if not TTS_ENABLED or tts_instance is None:
        return jsonify({"error": "TTS service not available."}), 503

    needs_fg_synthesis = False
    fg_lock_flag_in_session = f"_fg_synthesizing_chunk_{chunk_index}_"
    bg_lock_flag_in_session = f"_bg_synthesizing_chunk_{chunk_index}_" 

    with session_modification_lock:
        session_data = SESSIONS.get(session_id)
        if not session_data: return jsonify({"error": "Session not found"}), 404
        if not 0 <= chunk_index < len(session_data.get("text_chunks",[])):
            return jsonify({"error": "Invalid chunk index"}), 404

        current_status = session_data["synthesis_status"][chunk_index]
        fg_locked_by_another = session_data.get(fg_lock_flag_in_session, False)
        bg_locked = session_data.get(bg_lock_flag_in_session, False)
        
        logger.info(f"[FG_GetChunk] S:{session_id} C:{chunk_index}. Status: {current_status}. FG_lock:{fg_locked_by_another}, BG_lock:{bg_locked}")

        if current_status == "pending":
            if fg_locked_by_another or bg_locked:
                 logger.warning(f"[FG_GetChunk] S:{session_id} C:{chunk_index} - Pending but already locked (FG:{fg_locked_by_another}, BG:{bg_locked}). Returning busy.")
                 return jsonify({"status": "busy", "message": "This chunk is already being synthesized. Please try again shortly."}), 429
            session_data[fg_lock_flag_in_session] = True # Acquire FG lock
            session_data["synthesis_status"][chunk_index] = "synthesizing"
            needs_fg_synthesis = True
            logger.info(f"[FG_GetChunk] S:{session_id} C:{chunk_index} - Pending. FG lock acquired. Marked for FG synthesis.")
        elif current_status == "synthesizing":
            # If it's 'synthesizing', it means either this FG request (if needs_fg_synthesis becomes true later)
            # or a BG task, or another concurrent FG task holds the responsibility.
            logger.info(f"[FG_GetChunk] S:{session_id} C:{chunk_index} - Status 'synthesizing'. Returning busy/synthesizing.")
            return jsonify({"status": "synthesizing", "message": "Audio chunk is currently being generated. Retry shortly."}), 202
        elif current_status == "error":
            error_detail_msg = session_data["chunk_audio_filenames"][chunk_index] if isinstance(session_data["chunk_audio_filenames"][chunk_index], str) else "Previously failed"
            logger.warning(f"[FG_GetChunk] S:{session_id} C:{chunk_index} - Previously errored: {error_detail_msg}")
            return jsonify({"error": f"TTS synthesis previously failed for this chunk.", "details": error_detail_msg, "chunk_status": "error"}), 500
        elif current_status == "done": pass 
        else: # Should not happen
            logger.error(f"[FG_GetChunk] S:{session_id} C:{chunk_index} - Unknown status '{current_status}'.")
            return jsonify({"error": f"Unknown status '{current_status}' for chunk."}), 500

    if needs_fg_synthesis:
        logger.info(f"[FG_Synth_Call] S:{session_id} C:{chunk_index} - Initiating FG synthesis.")
        synthesis_success_fg, synthesis_error_msg_fg = _do_synthesize_chunk_with_subsplitting(session_id, chunk_index, is_background_thread=False)
        
        with session_modification_lock:
            session_data = SESSIONS.get(session_id) 
            if session_data and 0 <= chunk_index < len(session_data["synthesis_status"]):
                session_data.pop(fg_lock_flag_in_session, None) # Release FG lock
                logger.info(f"[FG_Synth_Done] S:{session_id} C:{chunk_index} - FG synthesis finished. Success: {synthesis_success_fg}. FG lock released.")
                if not synthesis_success_fg:
                    # _do_synthesize_chunk_with_subsplitting should set status to 'error'
                    # This is a fallback if it somehow didn't.
                    if session_data["synthesis_status"][chunk_index] != "error":
                        logger.error(f"[FG_Synth_Done] S:{session_id} C:{chunk_index} - FG synth failed but status was '{session_data['synthesis_status'][chunk_index]}'. Correcting. Error: {synthesis_error_msg_fg}")
                        session_data["synthesis_status"][chunk_index] = "error"
                        session_data["chunk_audio_filenames"][chunk_index] = f"fg_synth_failed_unspec: {synthesis_error_msg_fg or 'Unknown FG failure'}"
                    return jsonify({"error": "TTS synthesis failed for this chunk", "details": synthesis_error_msg_fg, "chunk_status": "error"}), 500
            elif not session_data:
                 logger.error(f"[FG_Synth_Done] S:{session_id} C:{chunk_index} - Session disappeared after FG synth.")
                 return jsonify({"error": "Session disappeared after synthesis attempt"}), 500
            # If chunk_index became invalid, error will be caught before serving.

    # Re-check status and other session data elements under lock
    with session_modification_lock:
        session_data = SESSIONS.get(session_id)
        if not session_data:
            logger.error(f"[FG_Serve] S:{session_id} C:{chunk_index} - Session disappeared before serving.")
            return jsonify({"error": "Session disappeared before serving chunk"}), 500
        
        num_text_chunks = len(session_data.get("text_chunks", []))
        if not (0 <= chunk_index < num_text_chunks and \
                0 <= chunk_index < len(session_data.get("synthesis_status", [])) and \
                0 <= chunk_index < len(session_data.get("chunk_audio_filenames", []))):
            logger.error(f"[FG_Serve] S:{session_id} C:{chunk_index} - Index out of bounds. Num text_chunks: {num_text_chunks}")
            return jsonify({"error": "Chunk index became invalid for session before serving"}), 500
        
        final_status = session_data["synthesis_status"][chunk_index]
        text_for_display = session_data["text_chunks"][chunk_index]

    if final_status == "done":
        audio_filename = session_data["chunk_audio_filenames"][chunk_index]
        # Check for known error strings in filename which might indicate inconsistent state
        if not audio_filename or not isinstance(audio_filename, str) or \
           any(err_str in audio_filename for err_str in ["sub_chunk_synth_failed", "empty_or_too_short", "ffmpeg_concat_failed", "audio_rename_failed", "bg_thread_wrapper_error", "done_status_inconsistent_filename"]):
            logger.error(f"[FG_Serve] S:{session_id} C:{chunk_index} - Status 'done' but filename problematic: '{audio_filename}'. Correcting status.")
            with session_modification_lock: 
                session_data_fix = SESSIONS.get(session_id)
                if session_data_fix and 0 <= chunk_index < len(session_data_fix["synthesis_status"]):
                    session_data_fix["synthesis_status"][chunk_index] = "error"
                    session_data_fix["chunk_audio_filenames"][chunk_index] = f"done_status_inconsistent_filename: {audio_filename or 'None'}"
            return jsonify({"error": "Chunk marked done but audio filename is invalid.", "details": "Internal error: inconsistent filename.", "chunk_status": "error"}), 500

        MAX_PREFETCH = 1 # Temporarily 1 for easier debugging
        for i in range(1, MAX_PREFETCH + 1):
            next_idx_to_synth = chunk_index + i
            next_chunk_bg_lock_flag = f"_bg_synthesizing_chunk_{next_idx_to_synth}_"
            next_chunk_fg_lock_flag = f"_fg_synthesizing_chunk_{next_idx_to_synth}_"

            with session_modification_lock:
                session_data_for_prefetch = SESSIONS.get(session_id)
                if not session_data_for_prefetch: break
                if next_idx_to_synth < len(session_data_for_prefetch["text_chunks"]) and \
                   session_data_for_prefetch["synthesis_status"][next_idx_to_synth] == "pending" and \
                   not session_data_for_prefetch.get(next_chunk_bg_lock_flag, False) and \
                   not session_data_for_prefetch.get(next_chunk_fg_lock_flag, False):
                    logger.info(f"[FG_Serve_Prefetch] S:{session_id} - Spawning BG thread for C:{next_idx_to_synth}")
                    thread = threading.Thread(target=synthesize_chunk_in_background, args=(session_id, next_idx_to_synth), name=f"BGsynth-{session_id}-{next_idx_to_synth}")
                    thread.daemon = True
                    thread.start()
                elif next_idx_to_synth < len(session_data_for_prefetch["text_chunks"]):
                     logger.debug(f"[FG_Serve_Prefetch] S:{session_id} - Not prefetching C:{next_idx_to_synth}. Status: {session_data_for_prefetch['synthesis_status'][next_idx_to_synth]}, BG_Lock: {session_data_for_prefetch.get(next_chunk_bg_lock_flag, False)}, FG_Lock: {session_data_for_prefetch.get(next_chunk_fg_lock_flag, False)}")
        
        logger.info(f"[FG_Serve] S:{session_id} C:{chunk_index} - Serving audio: {audio_filename}")
        return jsonify({
            "success": True, "audio_url": url_for('serve_chunk_audio', filename=audio_filename),
            "text_chunk": text_for_display, "chunk_index": chunk_index,
            "is_last_chunk": chunk_index == len(session_data.get("text_chunks",[])) - 1, # Use get for safety if session modified
            "chunk_status": "done"
        })

    logger.warning(f"[FG_Serve_End] S:{session_id} C:{chunk_index} - Unexpected final status: {final_status}")
    error_detail_final = "Unexpected error after processing."
    with session_modification_lock:
        session_data = SESSIONS.get(session_id)
        if session_data and 0 <= chunk_index < len(session_data.get("chunk_audio_filenames",[])) and isinstance(session_data["chunk_audio_filenames"][chunk_index], str):
            error_detail_final = session_data["chunk_audio_filenames"][chunk_index]
    return jsonify({"error": f"Unexpected final status '{final_status}' for chunk.", "details": error_detail_final, "chunk_status": final_status }), 500

@app.route('/retry_chunk_synthesis/<session_id>/<int:chunk_index>', methods=['POST'])
def retry_chunk_synthesis(session_id, chunk_index):
    if not TTS_ENABLED: return jsonify({"error": "TTS service not available.", "success": False}), 503
    with session_modification_lock:
        session_data = SESSIONS.get(session_id)
        if not session_data:
            logger.warning(f"[Retry] S:{session_id} C:{chunk_index} - Session not found.")
            return jsonify({"error": "Session not found", "success": False}), 404
        if not 0 <= chunk_index < len(session_data.get("text_chunks", [])):
            logger.warning(f"[Retry] S:{session_id} C:{chunk_index} - Invalid chunk index.")
            return jsonify({"error": "Invalid chunk index", "success": False}), 404

        current_status = session_data['synthesis_status'][chunk_index]
        logger.info(f"[Retry] S:{session_id} C:{chunk_index} - Manual retry requested. Current status: {current_status}")
        
        session_data["synthesis_status"][chunk_index] = "pending"
        session_data["chunk_audio_filenames"][chunk_index] = None
        session_data["synthesis_retry_count"][chunk_index] = 0 # Reset explicit retry count
        
        # Clear any locks associated with this chunk
        fg_lock_key = f"_fg_synthesizing_chunk_{chunk_index}_"
        bg_lock_key = f"_bg_synthesizing_chunk_{chunk_index}_"
        if session_data.pop(fg_lock_key, None):
            logger.info(f"[Retry] S:{session_id} C:{chunk_index} - Cleared FG lock.")
        if session_data.pop(bg_lock_key, None):
            logger.info(f"[Retry] S:{session_id} C:{chunk_index} - Cleared BG lock.")
            
        logger.info(f"[Retry] S:{session_id} C:{chunk_index} - Reset to 'pending'.")
    return jsonify({"success": True, "message": f"Chunk {chunk_index+1} marked for retry. Please request it again."})

@app.route('/static/chunk_audio/<path:filename>')
def serve_chunk_audio(filename):
    return send_from_directory(app.config['CHUNK_AUDIO_FOLDER'], filename)

@app.route('/static/<path:filename>')
def serve_general_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
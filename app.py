import os
import sqlite3
import tempfile
import subprocess
import numpy as np
import wave
import bcrypt
import streamlit as st
from io import BytesIO

# --- 1. CONFIGURATION & DATABASE SETUP ---
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}
DB_NAME = "users.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

# --- 2. AUTHENTICATION LOGIC ---
def register_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[1]):
        return user[0]
    return None

def lookup_user_by_id(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else "Unknown User"

# --- 3. CORE DSSS LOGIC (EMBEDDING & EXTRACTION) ---

def generate_pn_sequence(duration_samples):
    np.random.seed(42) # FIXED SEED: This is your 'Secret Key'
    return (np.random.randint(0, 2, duration_samples) * 2 - 1).astype(np.float64)

def embed_watermark_dsss(input_wav, output_wav, user_id, alpha=0.015):
    with wave.open(input_wav, 'rb') as wav:
        params = wav.getparams()
        frames = wav.readframes(params.nframes)
        audio_samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)

    user_id_bits = [int(bit) for bit in format(user_id, '08b')]
    payload = [1] + user_id_bits # 1 Signature bit + 8 ID bits
    
    total_samples = len(audio_samples)
    spreading_factor = total_samples // len(payload)
    pn_sequence = generate_pn_sequence(total_samples)

    watermark_signal = np.zeros(total_samples)
    for i, bit in enumerate(payload):
        start, end = i * spreading_factor, (i + 1) * spreading_factor
        bit_val = 1 if bit == 1 else -1
        watermark_signal[start:end] = bit_val * pn_sequence[start:end]

    watermarked_audio = audio_samples + (alpha * watermark_signal * np.max(np.abs(audio_samples)))
    watermarked_audio = np.clip(watermarked_audio, -32768, 32767).astype(np.int16)
    
    with wave.open(output_wav, 'wb') as wav_out:
        wav_out.setparams(params)
        wav_out.writeframes(watermarked_audio.tobytes())

def extract_watermark_dsss(input_wav):
    try:
        with wave.open(input_wav, "rb") as wav:
            frames = wav.readframes(wav.getparams().nframes)
        watermarked_samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        
        payload_length = 9
        total_samples = len(watermarked_samples)
        spreading_factor = total_samples // payload_length
        pn_sequence = generate_pn_sequence(total_samples)
        
        extracted_bits = []
        correlation_values = []

        for i in range(payload_length):
            start, end = i * spreading_factor, (i + 1) * spreading_factor
            correlation = np.mean(watermarked_samples[start:end] * pn_sequence[start:end])
            correlation_values.append(correlation)
            extracted_bits.append(1 if correlation > 0 else 0)

        # Validate signature bit and decode ID
        if abs(correlation_values[0]) < 10.0: # Detection Threshold
            return None, 0
            
        binary_str = "".join(map(str, extracted_bits[1:]))
        return int(binary_str, 2), abs(correlation_values[0])
    except Exception:
        return None, 0

# --- 4. FFMPEG UTILITIES ---

def run_ffmpeg(cmd):
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg Error: {e.stderr.decode()}")
        return False
    return True

# --- 5. STREAMLIT UI ---

def main():
    st.set_page_config(page_title="Guardian DRM System", layout="wide")
    init_db()

    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    # --- Sidebar: Auth ---
    if st.session_state.user_id is None:
        st.sidebar.title("🔐 Access Control")
        auth_mode = st.sidebar.radio("Action", ["Login", "Register"])
        user = st.sidebar.text_input("Username")
        pw = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button(auth_mode):
            if auth_mode == "Register":
                if register_user(user, pw): st.success("Registered!")
                else: st.error("Username exists.")
            else:
                uid = login_user(user, pw)
                if uid:
                    st.session_state.user_id = uid
                    st.rerun()
                else: st.error("Invalid Login.")
        st.stop()

    # --- Dashboard (Logged In) ---
    st.sidebar.success(f"User ID: {st.session_state.user_id}")
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.rerun()

    menu = ["📥 Download Protected Content", "🔍 Anti-Piracy Detector"]
    choice = st.selectbox("Select Module", menu)

    # --- MODULE 1: EMBEDDING ---
    if choice == "📥 Download Protected Content":
        st.header("Watermark My Video")
        video_file = st.file_uploader("Upload Video to Protect", type=list(ALLOWED_EXTENSIONS))
        
        if video_file:
            if st.button("Generate Protected Copy"):
                with tempfile.TemporaryDirectory() as tmp:
                    v_path = os.path.join(tmp, "in.mp4")
                    a_path = os.path.join(tmp, "in.wav")
                    wa_path = os.path.join(tmp, "out.wav")
                    out_v = os.path.join(tmp, "protected.mp4")
                    
                    with open(v_path, "wb") as f: f.write(video_file.read())
                    
                    with st.spinner("Baking your User ID into the audio..."):
                        # Extract -> Embed -> Merge
                        run_ffmpeg(["ffmpeg", "-y", "-i", v_path, "-vn", "-acodec", "pcm_s16le", a_path])
                        embed_watermark_dsss(a_path, wa_path, st.session_state.user_id)
                        run_ffmpeg(["ffmpeg", "-y", "-i", v_path, "-i", wa_path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", out_v])
                    
                    with open(out_v, "rb") as f:
                        st.download_button("Download Secure Video", f.read(), file_name="secure_content.mp4")

    # --- MODULE 2: DETECTION ---
    else:
        st.header("Identify Leaked Content")
        detect_file = st.file_uploader("Upload Suspected Video", type=list(ALLOWED_EXTENSIONS))
        
        if detect_file and st.button("Scan for Watermark"):
            with tempfile.TemporaryDirectory() as tmp:
                v_path = os.path.join(tmp, "detect.mp4")
                a_path = os.path.join(tmp, "detect.wav")
                with open(v_path, "wb") as f: f.write(detect_file.read())
                
                run_ffmpeg(["ffmpeg", "-y", "-i", v_path, "-vn", "-acodec", "pcm_s16le", a_path])
                ext_id, confidence = extract_watermark_dsss(a_path)
                
                if ext_id is not None:
                    username = lookup_user_by_id(ext_id)
                    st.error(f"🚨 **Watermark Detected!**")
                    st.write(f"**Associated User ID:** `{ext_id}`")
                    st.write(f"**Username in Database:** `{username}`")
                    st.write(f"**Signal Confidence:** `{confidence:.2f}`")
                else:
                    st.success("No watermark found.")

if __name__ == "__main__":
    main()
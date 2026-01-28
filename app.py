import os
import sqlite3
import tempfile
import subprocess
import numpy as np
import wave
import bcrypt
import streamlit as st

# --- 1. SETUP & DIRECTORIES ---
DB_NAME = "guardian.db"
UPLOAD_DIR = "master_videos"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)')
    # Table to track videos uploaded to the shared library
    c.execute('CREATE TABLE IF NOT EXISTS videos (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, uploader_id INTEGER)')
    conn.commit()
    conn.close()

# --- 2. DSSS CORE LOGIC ---
def generate_pn_sequence(duration_samples):
    np.random.seed(42) # Your secret key
    return (np.random.randint(0, 2, duration_samples) * 2 - 1).astype(np.float64)

def embed_watermark(input_wav, output_wav, user_id):
    with wave.open(input_wav, 'rb') as wav:
        params, frames = wav.getparams(), wav.readframes(wav.getparams().nframes)
        audio_samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)

    # Convert ID to 8 bits
    bits = [1] + [int(b) for b in format(user_id, '08b')]
    total_samples = len(audio_samples)
    sf = total_samples // len(bits)
    pn = generate_pn_sequence(total_samples)

    watermark = np.zeros(total_samples)
    for i, bit in enumerate(bits):
        val = 1 if bit == 1 else -1
        watermark[i*sf : (i+1)*sf] = val * pn[i*sf : (i+1)*sf]

    # Mix and Save (alpha 0.015)
    result = np.clip(audio_samples + (0.015 * watermark * np.max(np.abs(audio_samples))), -32768, 32767).astype(np.int16)
    with wave.open(output_wav, 'wb') as out:
        out.setparams(params)
        out.writeframes(result.tobytes())

# --- 3. UI HELPERS ---
def run_ffmpeg(cmd):
    subprocess.run(cmd, check=True, capture_output=True)

# --- 4. MAIN APP ---
def main():
    st.set_page_config(page_title="Anti-Piracy Portal", layout="wide")
    init_db()

    if 'uid' not in st.session_state: st.session_state.uid = None

    # --- LOGIN / REGISTER ---
    if st.session_state.uid is None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Login")
            u = st.text_input("Username", key="l_u")
            p = st.text_input("Password", type="password", key="l_p")
            if st.button("Login"):
                conn = sqlite3.connect(DB_NAME)
                res = conn.execute("SELECT id, password FROM users WHERE username=?", (u,)).fetchone()
                if res and bcrypt.checkpw(p.encode(), res[1]):
                    st.session_state.uid = res[0]
                    st.rerun()
                else: st.error("Failed")
        with col2:
            st.subheader("Register")
            nu = st.text_input("New Username")
            npw = st.text_input("New Password", type="password")
            if st.button("Create Account"):
                h = bcrypt.hashpw(npw.encode(), bcrypt.gensalt())
                try:
                    conn = sqlite3.connect(DB_NAME); conn.execute("INSERT INTO users (username, password) VALUES (?,?)", (nu, h))
                    conn.commit(); st.success("Done! Log in now.")
                except: st.error("User exists.")
        st.stop()

    # --- LOGGED IN UI ---
    st.sidebar.title(f"User ID: {st.session_state.uid}")
    if st.sidebar.button("Logout"):
        st.session_state.uid = None
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["📚 Shared Library", "📤 Upload Content", "🔍 Forensic Detector"])

    # TAB 2: UPLOAD
    with tab3:
        st.header("Forensic Detector")
        # (Insert the detector code here - same as your original snippet)
        st.info("Upload a leaked video here to find who it belongs to.")

    with tab2:
        st.header("Upload to Library")
        up_file = st.file_uploader("Select Master Video", type=['mp4','mkv','mov'])
        if up_file and st.button("Confirm Upload"):
            path = os.path.join(UPLOAD_DIR, up_file.name)
            with open(path, "wb") as f: f.write(up_file.read())
            conn = sqlite3.connect(DB_NAME)
            conn.execute("INSERT INTO videos (filename, uploader_id) VALUES (?,?)", (up_file.name, st.session_state.uid))
            conn.commit()
            st.success("File added to shared gallery!")

    # TAB 1: DOWNLOAD & WATERMARK
    with tab1:
        st.header("Available Content")
        conn = sqlite3.connect(DB_NAME)
        vids = conn.execute("SELECT filename FROM videos").fetchall()
        if not vids:
            st.warning("No videos available yet.")
        for v in vids:
            fname = v[0]
            with st.container():
                st.write(f"🎬 **{fname}**")
                if st.button(f"Download Watermarked Copy", key=fname):
                    with st.spinner("Injecting your unique ID..."):
                        with tempfile.TemporaryDirectory() as tmp:
                            in_v = os.path.join(UPLOAD_DIR, fname)
                            in_a, out_a, out_v = os.path.join(tmp,"1.wav"), os.path.join(tmp,"2.wav"), os.path.join(tmp,"out.mp4")
                            
                            # Processing
                            run_ffmpeg(["ffmpeg","-y","-i",in_v,"-vn","-acodec","pcm_s16le",in_a])
                            embed_watermark(in_a, out_a, st.session_state.uid)
                            run_ffmpeg(["ffmpeg","-y","-i",in_v,"-i",out_a,"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac",out_v])
                            
                            with open(out_v, "rb") as f:
                                st.download_button("Click to Download", f.read(), file_name=f"protected_{fname}")

if __name__ == "__main__":
    main()

"""
# REQUIRED LIBRARIES:
# Run this command in your terminal or Colab cell before starting the Streamlit app:
# pip install streamlit numpy scipy requests pedalboard

"""

import streamlit as st
import numpy as np
import scipy.io.wavfile as scipy_wav
import io
import requests
import uuid
import pedalboard

# --- CONFIGURATION & DATA ---
FEATURES = {
    "Bitcrush": {
        "_info": "Imagine a high-definition photo being turned into a pixelated mosaic. Bitcrushing reduces the 'detail' of the audio wave, creating a crunchy, robotic, or 'retro-gaming' sound.",
        "bit_depth": "Think of this as the resolution. 16 or 24 is clear and professional; 8 sounds like an old Nintendo; 2 or 4 sounds like a walkie-talkie from a storm.",
        "bit_depth_min": 1.0,
        "bit_depth_max": 24.0,
        "bit_depth_default": 8
    },
    "Convolution": {
        "_info": "Uses a 'fingerprint' of a real space (like a church) and applies it to your audio. Requires an Impulse Response (IR) file.",
        "mix": "How much of the 'fake room' you hear.",
        "mix_min": 0.0,
        "mix_max": 1.0,
        "mix_default": 1.0,
        "sample_rate": "The technical speed of the audio file.",
        "sample_rate_min": 8000.0,
        "sample_rate_max": 192000.0,
        "sample_rate_default": None
    }
}

# --- CSS INJECTION: HORIZONTAL LAYOUT & FAT SCROLLBARS ---
st.markdown("""
<style>
    /* Force Streamlit columns to act as a horizontally scrolling flex container */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        padding-bottom: 30px !important;
        align-items: flex-start !important;
    }
    
    /* Ensure each track column is wide enough to comfortably fit the sliders */
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        min-width: 500px !important;
        flex: 0 0 auto !important;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-right: 20px;
    }

    /* THE FAT SCROLLBAR */
    [data-testid="stHorizontalBlock"]::-webkit-scrollbar {
        height: 24px; /* Double standard size for fat fingers */
    }
    [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 12px;
    }
    [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb {
        background: #ff4b4b; /* Streamlit red for high visibility */
        border-radius: 12px;
        border: 3px solid rgba(0, 0, 0, 0.1); /* Creates a nice padding effect */
    }
    [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb:hover {
        background: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)


# --- STATE INITIALIZATION ---
if "tracks" not in st.session_state:
    st.session_state.tracks = {}

# --- CORE AUDIO FUNCTIONS ---
def fetch_and_decode_audio(url):
    """Downloads a WAV file and converts it safely to float32 for Pedalboard."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        audio_bytes = io.BytesIO(response.content)
        sample_rate, data = scipy_wav.read(audio_bytes)
        
        # SciPy loads as (samples, channels). Pedalboard wants (channels, samples).
        if data.ndim == 2:
            data = data.T
            
        # Lossless conversion to float32 (Pedalboard requirement)
        if data.dtype != np.float32:
            if data.dtype == np.int16:
                data = (data / 32768.0).astype(np.float32)
            elif data.dtype == np.int32:
                data = (data / 2147483648.0).astype(np.float32)
            elif data.dtype == np.uint8:
                data = ((data - 128) / 128.0).astype(np.float32)
            else:
                data = data.astype(np.float32)
                
        return data, sample_rate
    except Exception as e:
        st.error(f"Failed to load audio from {url}: {e}")
        return None, None

def create_playback_buffer(audio_array, sample_rate):
    """Converts an in-memory float32 array into a temporary WAV buffer for st.audio()"""
    buffer = io.BytesIO()
    # Transpose back from Pedalboard format (channels, samples) to SciPy format (samples, channels)
    write_data = audio_array.T if audio_array.ndim == 2 else audio_array
    scipy_wav.write(buffer, sample_rate, write_data)
    buffer.seek(0)
    return buffer

def add_new_track(name, url):
    """Creates a new track container and fetches the initial audio."""
    with st.spinner(f"Loading {name}..."):
        raw_audio, sr = fetch_and_decode_audio(url)
        if raw_audio is not None:
            track_id = str(uuid.uuid4())
            st.session_state.tracks[track_id] = {
                "name": name,
                "raw_audio": raw_audio,
                "sample_rate": sr,
                "chain": [] # Independent effect chain!
            }

def load_presets():
    """Downloads a batch of preset files from GitHub."""
    # ADD URL HERE: Replace these with your actual GitHub raw URLs
    preset_urls = [
        'https://github.com/ssopic/voice_pedal/blob/main/bark_generated_voices/v2_en_speaker_0.wav', 
        'https://github.com/ssopic/voice_pedal/blob/main/bark_generated_voices/v2_en_speaker_1.wav', 
        'https://github.com/ssopic/voice_pedal/blob/main/bark_generated_voices/v2_en_speaker_2.wav',
        'https://github.com/ssopic/voice_pedal/blob/main/bark_generated_voices/v2_en_speaker_3.wav', 
        'https://github.com/ssopic/voice_pedal/blob/main/bark_generated_voices/v2_en_speaker_4.wav', 
        'https://github.com/ssopic/voice_pedal/blob/main/bark_generated_voices/v2_en_speaker_5.wav', 
        'https://github.com/ssopic/voice_pedal/blob/main/bark_generated_voices/v2_en_speaker_6.wav', 
        'https://github.com/ssopic/voice_pedal/blob/main/bark_generated_voices/v2_en_speaker_7.wav', 
        'https://github.com/ssopic/voice_pedal/blob/main/bark_generated_voices/v2_en_speaker_8.wav', 
        'https://github.com/ssopic/voice_pedal/blob/main/bark_generated_voices/v2_en_speaker_9.wav']
    for idx, url in enumerate(preset_urls):
        add_new_track(f"Preset Voice {idx+1}", url)

def replace_track_audio(track_id, url):
    """Replaces the audio of a track WITHOUT deleting the effect chain."""
    with st.spinner("Replacing audio..."):
        raw_audio, sr = fetch_and_decode_audio(url)
        if raw_audio is not None:
            st.session_state.tracks[track_id]["raw_audio"] = raw_audio
            st.session_state.tracks[track_id]["sample_rate"] = sr


# --- CHAIN MANAGEMENT FUNCTIONS ---
def add_effect_to_track(track_id, effect_name):
    feature_data = FEATURES[effect_name]
    params = {}
    for key, value in feature_data.items():
        if key.endswith('_default'):
            param_name = key.replace('_default', '')
            safe_value = value if value is not None else feature_data.get(f"{param_name}_min", 0.0)
            params[param_name] = safe_value

    st.session_state.tracks[track_id]["chain"].append({
        "id": str(uuid.uuid4()),
        "name": effect_name,
        "values": params
    })

def move_effect(track_id, idx, direction):
    chain = st.session_state.tracks[track_id]["chain"]
    new_idx = idx + direction
    if 0 <= new_idx < len(chain):
        chain[idx], chain[new_idx] = chain[new_idx], chain[idx]


# --- UI: TOP CONTROLS ---
st.set_page_config(layout="wide", page_title="Multi-Track Audio Lab")
st.title("🎛️ Multi-Track Audio Lab")
st.caption("Load tracks, build independent effect chains, and audition them step-by-step.")

top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        custom_url = st.text_input("Load Custom WAV URL", placeholder="https://example.com/audio.wav")
    with col_b:
        st.write("") # Spacer
        if st.button("Download & Add Track", use_container_width=True):
            if custom_url:
                add_new_track(f"Custom Track {len(st.session_state.tracks) + 1}", custom_url)
with top_col2:
    st.write("") # Spacer
    if st.button("📥 Load Preset Voices", type="primary", use_container_width=True):
        load_presets()

st.divider()

# --- UI: HORIZONTAL TRACK LAYOUT ---
if not st.session_state.tracks:
    st.info("No audio tracks loaded. Use the controls above to fetch some WAV files!")
else:
    # Streamlit natively puts these in a flex container. Our CSS above forces them to scroll horizontally!
    cols = st.columns(len(st.session_state.tracks))
    
    for col, (track_id, track) in zip(cols, st.session_state.tracks.items()):
        with col:
            # TRACK HEADER
            head_c1, head_c2 = st.columns([3, 1])
            head_c1.subheader(track["name"])
            if head_c2.button("✖ Remove", key=f"del_track_{track_id}"):
                del st.session_state.tracks[track_id]
                st.rerun()
                
            # REPLACE AUDIO
            replace_url = st.text_input("Replace Voice URL", key=f"rep_{track_id}", placeholder="New WAV URL...")
            if replace_url:
                if st.button("Replace (Keep Effects)", key=f"rep_btn_{track_id}"):
                    replace_track_audio(track_id, replace_url)
                    st.rerun()

            st.write("**Original Raw Audio:**")
            st.audio(create_playback_buffer(track["raw_audio"], track["sample_rate"]), format="audio/wav")
            
            # EFFECT ADDER
            st.write("---")
            add_col1, add_col2 = st.columns([2, 1])
            selected_eff = add_col1.selectbox("Choose Effect", options=FEATURES.keys(), key=f"sel_{track_id}")
            if add_col2.button("➕ Add", key=f"add_{track_id}", use_container_width=True):
                add_effect_to_track(track_id, selected_eff)
                st.rerun()
            
            # -----------------------------------------------------------------
            # THE AUDIO PROCESSING ENGINE & STEP-BY-STEP UI
            # -----------------------------------------------------------------
            current_audio_array = track["raw_audio"]
            sr = track["sample_rate"]
            
            for e_idx, effect in enumerate(track["chain"]):
                st.write("") # Spacer
                with st.container(border=True):
                    
                    # Effect Header & Controls
                    c_title, c_up, c_down, c_del = st.columns([4, 1, 1, 1])
                    c_title.markdown(f"**#{e_idx + 1} {effect['name']}**")
                    if c_up.button("↑", key=f"up_{effect['id']}"):
                        move_effect(track_id, e_idx, -1)
                        st.rerun()
                    if c_down.button("↓", key=f"down_{effect['id']}"):
                        move_effect(track_id, e_idx, 1)
                        st.rerun()
                    if c_del.button("✖", key=f"del_{effect['id']}"):
                        st.session_state.tracks[track_id]["chain"].pop(e_idx)
                        st.rerun()
                        
                    # Effect Sliders
                    feature_data = FEATURES[effect['name']]
                    for p_name, p_val in effect["values"].items():
                        min_val = float(feature_data.get(f"{p_name}_min", 0.0))
                        max_val = float(feature_data.get(f"{p_name}_max", 1.0))
                        
                        new_val = st.slider(
                            label=p_name.replace("_", " ").title(),
                            min_value=min_val,
                            max_value=max_val,
                            value=float(p_val),
                            key=f"sl_{effect['id']}_{p_name}",
                            help=feature_data.get(p_name, "")
                        )
                        effect["values"][p_name] = new_val
                    
                    # --- LOSSLESS IN-MEMORY PROCESSING ---
                    try:
                        # 1. Fetch the exact effect class from Pedalboard
                        effect_class = getattr(pedalboard, effect['name'])
                        
                        # 2. Instantiate it with our UI values
                        pb_effect = effect_class(**effect["values"])
                        board = pedalboard.Pedalboard([pb_effect])
                        
                        # 3. Process the audio array from the *previous* step
                        current_audio_array = board(current_audio_array, sr)
                        
                        # 4. Create a temporary buffer so the user can hear this exact step
                        st.caption(f"🔊 Listen after {effect['name']}:")
                        st.audio(create_playback_buffer(current_audio_array, sr), format="audio/wav")
                        
                    except Exception as e:
                        st.error(f"Error processing {effect['name']}: {e}")
            
            # JSON DICTIONARY DUMP
            st.write("---")
            st.subheader("Effect Dictionary")
            final_output = []
            for i, item in enumerate(track["chain"]):
                final_output.append({
                    "order": i + 1,
                    "effect": item["name"],
                    "type": f"pedalboard_native.{item['name']}",
                    "params": item["values"]
                })
            st.json(final_output)

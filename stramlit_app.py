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
# COMPREHENSIVE PEDALBOARD EDUCATIONAL HINTS WITH CONSTRAINTS AND DEFAULTS
# This dictionary maps built-in Pedalboard effects to beginner-friendly explanations, safe ranges, and default values.

FEATURES = {
    "Bitcrush": {
        "_info": "Imagine a high-definition photo being turned into a pixelated mosaic. Bitcrushing reduces the 'detail' of the audio wave, creating a crunchy, robotic, or 'retro-gaming' sound.",
        "bit_depth": "Think of this as the resolution. 16 or 24 is clear and professional; 8 sounds like an old Nintendo; 2 or 4 sounds like a walkie-talkie from a storm.",
        "bit_depth_min": 1.0,
        "bit_depth_max": 24.0,
        "bit_depth_default": 8
    },
    "Chorus": {
        "_info": "This makes one instrument sound like a group. It slightly detunes and delays a copy of your sound to create thickness and shimmer.",
        "rate_hz": "How fast the sound 'wobbles'. Low values create a slow wave; high values create a fast, dizzying shimmer.",
        "rate_hz_min": 0.01,
        "rate_hz_max": 100.0,
        "rate_hz_default": 1.0,
        "depth": "How 'out of tune' the voices get. High values make the effect very obvious and 'underwater' sounding.",
        "depth_min": 0.0,
        "depth_max": 1.0,
        "depth_default": 0.25,
        "centre_delay_ms": "The base delay between the voices. Small changes here affect how 'thick' the group sounds.",
        "centre_delay_ms_min": 1.0,
        "centre_delay_ms_max": 100.0,
        "centre_delay_ms_default": 7.0,
        "feedback": "Feeds the sound back into itself. This can add a metallic edge to the chorus.",
        "feedback_min": 0.0,
        "feedback_max": 0.95,
        "feedback_default": 0.0,
        "mix": "0 is just the original sound; 1.0 is full chorus. 0.5 is the 'sweet spot' for most users.",
        "mix_min": 0.0,
        "mix_max": 1.0,
        "mix_default": 0.5
    },
    "Clipping": {
        "_info": "This mimics what happens when you push a speaker too hard. It 'chops off' the tops of the sound waves, creating a harsh, aggressive distortion.",
        "threshold_db": "The level where the chopping starts. Lowering this value makes the sound more distorted.",
        "threshold_db_min": -60.0,
        "threshold_db_max": 0.0,
        "threshold_db_default": -6.0
    },
    "Compressor": {
        "_info": "The 'Auto-Volume' effect. It makes quiet sounds louder and loud sounds quieter, resulting in a consistent, professional-level energy.",
        "threshold_db": "The volume level where the compressor starts working.",
        "threshold_db_min": -60.0,
        "threshold_db_max": 0.0,
        "threshold_db_default": 0,
        "ratio": "How much the volume is turned down. 1:1 does nothing; 10:1 'squashes' the sound significantly.",
        "ratio_min": 1.0,
        "ratio_max": 50.0,
        "ratio_default": 1,
        "attack_ms": "How fast the compressor reacts. A slow attack lets the initial 'thump' through.",
        "attack_ms_min": 0.1,
        "attack_ms_max": 500.0,
        "attack_ms_default": 1.0,
        "release_ms": "How long it takes for the volume to return to normal after the sound drops.",
        "release_ms_min": 1.0,
        "release_ms_max": 3000.0,
        "release_ms_default": 100
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
    },
    "Delay": {
        "_info": "A classic echo. It repeats your sound after a set amount of time.",
        "delay_seconds": "The time between the original sound and the echo.",
        "delay_seconds_min": 0.01,
        "delay_seconds_max": 20.0,
        "delay_seconds_default": 0.5,
        "feedback": "How many echoes you hear. 0 is one echo; 0.9 is a long trail.",
        "feedback_min": 0.0,
        "feedback_max": 1.0,
        "feedback_default": 0.0,
        "mix": "How loud the echoes are compared to the original sound.",
        "mix_min": 0.0,
        "mix_max": 1.0,
        "mix_default": 0.5
    },
    "Distortion": {
        "_info": "Adds heat and grit, like an overdriven guitar amp.",
        "drive_db": "The 'heat'. Turn this up for more fuzz and grit.",
        "drive_db_min": 0.0,
        "drive_db_max": 50.0,
        "drive_db_default": 25
    },
    "Gain": {
        "_info": "A simple volume knob inside the digital signal path.",
        "gain_db": "Positive values make it louder; negative values make it quieter.",
        "gain_db_min": -60.0,
        "gain_db_max": 24.0,
        "gain_db_default": 1.0
    },
    "GSMFullRateCompressor": {
        "_info": "Mimics the sound of an old 2G cell phone call. It adds a specific type of digital 'lo-fi' crunch.",
        "quality": "Technical setting for the resampling quality.",
        "quality_min": 0,
        "quality_max": 10,
        "quality_default": 10  # Derived from "<Quality.WindowedSinc8: 10>"
    },
    "HighShelfFilter": {
        "_info": "The 'Treble' knob. It boosts or cuts all high-frequency sounds.",
        "cutoff_frequency_hz": "Where the treble range starts.",
        "cutoff_frequency_hz_min": 20.0,
        "cutoff_frequency_hz_max": 20000.0,
        "cutoff_frequency_hz_default": 440,
        "gain_db": "The amount of boost or cut.",
        "gain_db_min": -24.0,
        "gain_db_max": 24.0,
        "gain_db_default": 0.0,
        "q": "The sharpness of the curve.",
        "q_min": 0.1,
        "q_max": 10.0,
        "q_default": 0.7071067690849304
    },
    "HighpassFilter": {
        "_info": "Cuts out the 'bass mud' and let high sounds pass through.",
        "cutoff_frequency_hz": "Everything below this number is removed.",
        "cutoff_frequency_hz_min": 20.0,
        "cutoff_frequency_hz_max": 20000.0,
        "cutoff_frequency_hz_default": 50
    },
    "LadderFilter": {
        "_info": "A classic synth filter with a warm, vintage character.",
        "cutoff_hz": "The point where the filter acts. Moving this creates 'wah' sounds.",
        "cutoff_hz_min": 20.0,
        "cutoff_hz_max": 20000.0,
        "cutoff_hz_default": 200,
        "resonance": "Adds a whistle or peak at the cutoff frequency.",
        "resonance_min": 0.0,
        "resonance_max": 1.0,
        "resonance_default": 0,
        "drive": "Internal saturation for a beefier sound.",
        "drive_min": 1.0,
        "drive_max": 10.0,
        "drive_default": 1.0
    },
    "Limiter": {
        "_info": "A hard ceiling that prevents audio from ever getting too loud (clipping).",
        "threshold_db": "The maximum volume allowed.",
        "threshold_db_min": -60.0,
        "threshold_db_max": 0.0,
        "threshold_db_default": -10.0,
        "release_ms": "How fast the limiter resets after a peak.",
        "release_ms_min": 1.0,
        "release_ms_max": 1000.0,
        "release_ms_default": 100.0
    },
    "LowShelfFilter": {
        "_info": "The 'Bass' knob. Boosts or cuts the low frequencies.",
        "cutoff_frequency_hz": "Where the bass range ends.",
        "cutoff_frequency_hz_min": 20.0,
        "cutoff_frequency_hz_max": 20000.0,
        "cutoff_frequency_hz_default": 440,
        "gain_db": "The amount of bass boost or cut.",
        "gain_db_min": -24.0,
        "gain_db_max": 24.0,
        "gain_db_default": 0.0,
        "q": "The sharpness of the curve.",
        "q_min": 0.1,
        "q_max": 10.0,
        "q_default": 0.7071067690849304
    },
    "LowpassFilter": {
        "_info": "The 'Muffler'. Cuts out high frequencies to make things sound dark or distant.",
        "cutoff_frequency_hz": "Everything above this number is removed.",
        "cutoff_frequency_hz_min": 20.0,
        "cutoff_frequency_hz_max": 20000.0,
        "cutoff_frequency_hz_default": 50
    },
    "MP3Compressor": {
        "_info": "Simulates the sound of a low-quality MP3 file, adding 'watery' artifacts.",
        "vbr_quality": "Lower is better quality; higher is more compressed/broken sounding.",
        "vbr_quality_min": 0.0,
        "vbr_quality_max": 9.0,
        "vbr_quality_default": 2.0
    },
    "NoiseGate": {
        "_info": "Cuts off sound when it gets too quiet, removing background noise.",
        "threshold_db": "The level where the gate closes.",
        "threshold_db_min": -100.0,
        "threshold_db_max": 0.0,
        "threshold_db_default": -100.0,
        "ratio": "How hard the audio is cut when the gate is closed.",
        "ratio_min": 1.0,
        "ratio_max": 50.0,
        "ratio_default": 10,
        "attack_ms": "How fast the gate opens.",
        "attack_ms_min": 0.1,
        "attack_ms_max": 500.0,
        "attack_ms_default": 1.0,
        "release_ms": "How fast the gate closes.",
        "release_ms_min": 1.0,
        "release_ms_max": 3000.0,
        "release_ms_default": 100.0
    },
    "PeakFilter": {
        "_info": "A surgical tool to boost or cut a specific frequency note.",
        "cutoff_frequency_hz": "The exact frequency note you are targeting.",
        "cutoff_frequency_hz_min": 20.0,
        "cutoff_frequency_hz_max": 20000.0,
        "cutoff_frequency_hz_default": 440,
        "gain_db": "The boost or cut at that specific spot.",
        "gain_db_min": -24.0,
        "gain_db_max": 24.0,
        "gain_db_default": 0.0,
        "q": "How narrow the focus is. High Q is a tiny needle; low Q is a wide brush.",
        "q_min": 0.1,
        "q_max": 20.0,
        "q_default": 0.7071067690849304
    },
    "Phaser": {
        "_info": "A swirling 'whoosh' effect caused by moving phase cancellations.",
        "rate_hz": "Speed of the swirling.",
        "rate_hz_min": 0.01,
        "rate_hz_max": 20.0,
        "rate_hz_default": 1.0,
        "depth": "Width of the swirl.",
        "depth_min": 0.0,
        "depth_max": 1.0,
        "depth_default": 0.5,
        "centre_frequency_hz": "The middle frequency of the swirl.",
        "centre_frequency_hz_min": 100.0,
        "centre_frequency_hz_max": 10000.0,
        "centre_frequency_hz_default": 1300.0,
        "feedback": "Adds a metallic resonance to the swirl.",
        "feedback_min": 0.0,
        "feedback_max": 0.99,
        "feedback_default": 0.0,
        "mix": "Dry/Wet balance.",
        "mix_min": 0.0,
        "mix_max": 1.0,
        "mix_default": 0.5
    },
    "PitchShift": {
        "_info": "Changes the musical pitch without changing the speed of the audio.",
        "semitones": "Steps in pitch. +12 is one octave up; -12 is one octave down.",
        "semitones_min": -24.0,
        "semitones_max": 24.0,
        "semitones_default": 0.0
    },
    "Resample": {
        "_info": "Changes the sample rate of the audio, often used for lo-fi effects.",
        "target_sample_rate": "The new frequency speed.",
        "target_sample_rate_min": 2000.0,
        "target_sample_rate_max": 192000.0,
        "target_sample_rate_default": 8000.0,
        "quality": "The mathematical precision of the resample.",
        "quality_min": 0,
        "quality_max": 10,
        "quality_default": 8  # Derived from "<Quality.WindowedSinc32: 8>"
    },
    "Reverb": {
        "_info": "Adds the sound of a physical room or space.",
        "room_size": "Size of the space (Closet vs Cathedral).",
        "room_size_min": 0.0,
        "room_size_max": 1.0,
        "room_size_default": 0.5,
        "damping": "How much high-frequency sound the walls absorb.",
        "damping_min": 0.0,
        "damping_max": 1.0,
        "damping_default": 0.5,
        "wet_level": "Volume of the echo part.",
        "wet_level_min": 0.0,
        "wet_level_max": 1.0,
        "wet_level_default": 0.33,
        "dry_level": "Volume of the clean part.",
        "dry_level_min": 0.0,
        "dry_level_max": 1.0,
        "dry_level_default": 0.4,
        "width": "Stereo width of the room.",
        "width_min": 0.0,
        "width_max": 1.0,
        "width_default": 1.0,
        "freeze_mode": "Holds the echo forever if set to 1.",
        "freeze_mode_min": 0.0,
        "freeze_mode_max": 1.0,
        "freeze_mode_default": 0.0
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

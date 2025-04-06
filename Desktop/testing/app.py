import numpy as np
np.complex = complex  # Fix for librosa compatibility

import os
import librosa
import json
import tempfile
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import speech_recognition as sr
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import traceback

app = Flask(__name__)
app.secret_key = "vocaledge_secret_key"

# Create necessary directories
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
REPORTS_FOLDER = os.path.join(os.getcwd(), 'reports')
STATIC_FOLDER = os.path.join(os.getcwd(), 'static')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER

# Audio stream config
RATE = 22050
RECORD_SECONDS = 20

# NLP Use-Case Detection
USE_CASE_KEYWORDS = {
    "interview": ["interview", "introduction", "self intro", "tell me about yourself"],
    "singing": ["song", "singing", "practice singing"],
    "public_speaking": ["speech", "presentation", "talk"]
}

USE_CASE_SUGGESTIONS = {
    "interview": [
        "Practice using a timer to simulate interview pressure.",
        "Use STAR format (Situation, Task, Action, Result) in responses.",
        "Keep answers concise and confident."
    ],
    "singing": [
        "Avoid dairy or cold items like ice cream before singing.",
        "Warm up your voice with humming or lip trills.",
        "Stay hydrated and avoid yelling before sessions."
    ],
    "public_speaking": [
        "Practice in front of a mirror or record yourself.",
        "Work on intonation and pace to maintain engagement.",
        "Use pauses strategically for emphasis."
    ]
}

def check_initial_silence(y, sr, threshold=0.01, duration_sec=5):
    check_samples = int(sr * duration_sec)
    energy = np.mean(np.abs(y[:check_samples]))
    if energy < threshold:
        return "You remained silent in the first few seconds. Try starting promptly."
    return None

def check_audio_presence(y):
    if np.max(np.abs(y)) < 0.005:
        return False
    return True

def detect_use_case_from_text(transcribed_text):
    transcribed_text = transcribed_text.lower()
    for use_case, keywords in USE_CASE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in transcribed_text:
                return use_case
    return None

def get_custom_suggestions(use_case):
    return USE_CASE_SUGGESTIONS.get(use_case, [])

def analyze_voice(y, rate):
    y = y / (np.max(np.abs(y)) + 1e-5)
    
    silence_warning = check_initial_silence(y, rate)
    if not check_audio_presence(y):
        return {
            "confidence_level": "No Voice",
            "confidence_score": 0,
            "suggestions": ["Please speak clearly and close to the mic."],
            "warnings": [silence_warning] if silence_warning else [],
            "pitch_mean": 0,
            "pitch_std": 0,
            "energy_mean": 0,
            "energy_std": 0,
            "pause_count": 0,
            "filler_count": 0
        }

    pitches, magnitudes = librosa.piptrack(y=y, sr=rate, fmin=80, fmax=600)
    pitch_values = []
    for t in range(magnitudes.shape[1]):
        index = np.argmax(magnitudes[:, t])
        pitch = pitches[index, t]
        if pitch > 50:
            pitch_values.append(pitch)

    if len(pitch_values) > 5:
        pitch_values = median_filter(pitch_values, size=5)
    pitch_std = np.std(pitch_values)
    pitch_mean = np.mean(pitch_values)

    energy = librosa.feature.rms(y=y)[0]
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)

    non_silent = librosa.effects.split(y, top_db=30)
    total_silence_duration = 0
    pause_count = 0
    for i in range(1, len(non_silent)):
        gap = (non_silent[i][0] - non_silent[i - 1][1]) / rate
        if gap > 0.25:
            pause_count += 1
            total_silence_duration += gap

    smoothed = np.convolve(np.abs(y), np.ones(1000)/1000, mode='valid')
    peaks, _ = find_peaks(smoothed, height=0.01, distance=1000)
    filler_count = len(peaks) // 30

    debug_info = {
        "pitch_mean": f"{pitch_mean:.1f} Hz",
        "pitch_std": f"{pitch_std:.2f}",
        "energy_mean": f"{energy_mean:.5f}",
        "energy_std": f"{energy_std:.5f}",
        "pauses": pause_count,
        "total_silence": f"{total_silence_duration:.2f}s",
        "filler_count": filler_count
    }

    normalized_pitch_std = np.clip(pitch_std / 50.0, 0, 1)
    normalized_energy = np.clip(energy_mean * 50, 0, 1)
    normalized_fillers = np.clip(filler_count / 10, 0, 1)
    normalized_pauses = np.clip(pause_count / 5, 0, 1)

    confidence_score = 100
    confidence_score -= normalized_pitch_std * 20
    confidence_score -= (1 - normalized_energy) * 25
    confidence_score -= normalized_fillers * 30
    confidence_score -= normalized_pauses * 25
    confidence_score = max(confidence_score, 0)

    if confidence_score >= 75:
        confidence_level = "Confident"
    elif confidence_score >= 50:
        confidence_level = "Moderate"
    else:
        confidence_level = "Needs Improvement"

    suggestions = []
    if pitch_std < 20:
        suggestions.append("Increase pitch variation to sound more engaging.")
    if energy_mean < 0.02:
        suggestions.append("Speak with more volume and energy.")
    if filler_count >= 3:
        suggestions.append("Practice reducing filler words like 'um' and 'uh'.")
    if pause_count >= 3:
        suggestions.append("Minimize long pauses for smoother delivery.")

    transcribed_text = ""
    try:
        r = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio = r.record(source)
        transcribed_text = r.recognize_google(audio)
        use_case = detect_use_case_from_text(transcribed_text)
        if use_case:
            suggestions += get_custom_suggestions(use_case)
    except Exception:
        pass

    warnings = []
    if silence_warning:
        warnings.append(silence_warning)

    return {
        "confidence_level": confidence_level,
        "confidence_score": round(confidence_score, 1),
        "suggestions": suggestions,
        "warnings": warnings,
        "pitch_mean": round(float(pitch_mean), 1),
        "pitch_std": round(float(pitch_std), 2),
        "energy_mean": float(energy_mean),
        "energy_std": float(energy_std),
        "pause_count": pause_count,
        "filler_count": filler_count,
        "debug_info": debug_info,
        "transcribed_text": transcribed_text
    }

def generate_spectrogram(y, sr, filename="spectrogram.png"):
    plt.figure(figsize=(12, 6))

    # Waveform (top)
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.7, color='mediumpurple')
    plt.title("Waveform")
    plt.ylabel("Amplitude")
    plt.xlabel("")

    # Spectrogram (bottom)
    plt.subplot(2, 1, 2)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram (Log Frequency Scale)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    
    # Save to static folder
    filepath = os.path.join(STATIC_FOLDER, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

def process_audio_file(file_path):
    try:
        # Try to load the audio file with librosa, which supports various formats
        y, sr = librosa.load(file_path, sr=RATE)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Save as WAV for speech recognition
        temp_wav_path = "temp.wav"
        sf.write(temp_wav_path, y, sr)
        
        # Analyze the voice
        analysis_results = analyze_voice(y, sr)
        
        # Generate spectrogram
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spec_filename = f"spectrogram_{timestamp}.png"
        spectrogram_path = generate_spectrogram(y, sr, spec_filename)
        analysis_results["spectrogram"] = spec_filename
        
        # Clean up temporary file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            
        return analysis_results, y, sr
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        print(traceback.format_exc())
        return {
            "error": f"Failed to process audio file: {str(e)}",
            "confidence_level": "Error",
            "confidence_score": 0,
            "suggestions": ["Technical error occurred. Please try again."],
            "warnings": [],
        }, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    
    # Process the audio file
    try:
        analysis_results, y, sr = process_audio_file(temp_path)
        session['last_analysis'] = analysis_results
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/spectrogram/<filename>')
def serve_spectrogram(filename):
    return send_file(os.path.join(STATIC_FOLDER, filename), mimetype='image/png')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

if __name__ == '__main__':
    app.run(debug=True)
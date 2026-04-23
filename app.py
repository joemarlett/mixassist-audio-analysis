import os
import io
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ============================================================
# HEALTH CHECK
# ============================================================
@app.route('/', methods=['GET'])
def health():
    return jsonify({ 'status': 'MixAssist Audio Analysis is live!', 'version': '2.0' })

# ============================================================
# ANALYZE AUDIO
# ============================================================
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'audio' not in request.files:
            return jsonify({ 'error': 'No audio file provided' }), 400

        file = request.files['audio']
        if not file.filename:
            return jsonify({ 'error': 'Empty filename' }), 400

        audio_bytes  = file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # ── Load stereo at native sample rate ─────────────────
        # sr=None preserves original sample rate (44.1k, 48k, 96k etc)
        # mono=False preserves stereo channels for accurate width/phase measurements
        try:
            audio_buffer.seek(0)
            y_stereo, sr = librosa.load(audio_buffer, mono=False, sr=None, duration=180)
        except Exception:
            audio_buffer.seek(0)
            y_stereo, sr = librosa.load(audio_buffer, mono=False, sr=44100, duration=180)

        # ── Build mono sum for frequency analysis ─────────────
        if y_stereo.ndim == 2:
            # True stereo file — keep both
            y_left  = y_stereo[0]
            y_right = y_stereo[1]
            y_mono  = (y_left + y_right) / 2.0
        else:
            # File was already mono
            y_mono  = y_stereo
            y_left  = y_stereo
            y_right = y_stereo

        if len(y_mono) == 0:
            return jsonify({ 'error': 'Could not decode audio file' }), 400

        result = extract_spectral_data(y_mono, y_left, y_right, sr)
        return jsonify(result)

    except Exception as e:
        print(f'Analysis error: {e}')
        return jsonify({ 'error': f'Analysis failed: {str(e)}' }), 500


# ============================================================
# SPECTRAL EXTRACTION
# ============================================================
def extract_spectral_data(y_mono, y_left, y_right, sr):

    # ── Frequency band energy (mono sum) ─────────────────────
    stft  = np.abs(librosa.stft(y_mono, n_fft=4096, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    def band_energy_db(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        if not np.any(mask):
            return -60.0
        band = stft[mask, :]
        rms  = np.sqrt(np.mean(band ** 2))
        if rms == 0:
            return -60.0
        return float(20 * np.log10(rms + 1e-10))

    sub_bass = band_energy_db(20,    60)
    bass     = band_energy_db(60,    120)
    low_mid  = band_energy_db(120,   300)
    mid      = band_energy_db(300,   2000)
    high_mid = band_energy_db(2000,  8000)
    air      = band_energy_db(8000,  20000)

    # ── RMS and Peak (mono) ───────────────────────────────────
    rms_linear = float(np.sqrt(np.mean(y_mono ** 2)))
    rms_db     = float(20 * np.log10(rms_linear + 1e-10))
    peak_db    = float(20 * np.log10(np.max(np.abs(y_mono)) + 1e-10))

    # ── Dynamic Range (mono) ──────────────────────────────────
    frame_length = sr // 4
    hop_length   = frame_length // 2
    frames       = librosa.util.frame(y_mono, frame_length=frame_length, hop_length=hop_length)
    frame_rms    = np.sqrt(np.mean(frames ** 2, axis=0))
    frame_rms_db = 20 * np.log10(frame_rms + 1e-10)
    p95          = float(np.percentile(frame_rms_db, 95))
    p10          = float(np.percentile(frame_rms_db, 10))
    dynamic_range = float(p95 - p10)

    # ── TRUE Stereo Width (L/R correlation) ───────────────────
    # Mid = L+R, Side = L-R
    # Width = Side energy / (Mid energy + Side energy)
    # 0 = pure mono, 1 = pure stereo/wide
    mid_signal  = y_left + y_right
    side_signal = y_left - y_right
    mid_energy  = float(np.mean(mid_signal ** 2))
    side_energy = float(np.mean(side_signal ** 2))
    total_energy = mid_energy + side_energy
    if total_energy < 1e-10:
        stereo_width = 0.0
    else:
        stereo_width = float(round(side_energy / total_energy, 3))

    # ── TRUE Mono Compatibility (L/R phase correlation) ───────
    # Pearson correlation between L and R
    # +1.0 = perfect mono compatibility
    # 0.0  = completely unrelated channels
    # -1.0 = completely out of phase (worst case)
    # Normalize to 0-1 for display (0 = bad, 1 = perfect)
    if np.std(y_left) < 1e-10 or np.std(y_right) < 1e-10:
        mono_compatibility = 1.0
    else:
        correlation = float(np.corrcoef(y_left, y_right)[0, 1])
        # Map from [-1, 1] to [0, 1]
        mono_compatibility = float(round((correlation + 1.0) / 2.0, 3))

    return {
        'sub_bass':           round(sub_bass,        2),
        'bass':               round(bass,             2),
        'low_mid':            round(low_mid,          2),
        'mid':                round(mid,              2),
        'high_mid':           round(high_mid,         2),
        'air':                round(air,              2),
        'rms':                round(rms_db,           2),
        'peak':               round(peak_db,          2),
        'dynamic_range':      round(dynamic_range,    2),
        'stereo_width':       round(stereo_width,     3),
        'mono_compatibility': round(mono_compatibility, 3),
        'duration_seconds':   round(len(y_mono) / sr, 1)
    }


# ============================================================
# COMPARE TWO ANALYSES
# ============================================================
@app.route('/compare', methods=['POST'])
def compare():
    try:
        data      = request.get_json()
        user_mix  = data.get('user_mix')
        reference = data.get('reference')

        if not user_mix or not reference:
            return jsonify({ 'error': 'user_mix and reference are required' }), 400

        report = build_comparison_report(user_mix, reference)
        return jsonify({ 'report': report })

    except Exception as e:
        print(f'Compare error: {e}')
        return jsonify({ 'error': f'Comparison failed: {str(e)}' }), 500


def build_comparison_report(user, ref):
    lines = []
    lines.append('MIX ANALYSIS COMPARISON REPORT\n')

    def delta_label(user_val, ref_val, band_name, unit='dB'):
        diff      = user_val - ref_val
        if abs(diff) < 0.5:
            return f'{band_name}: Within 0.5{unit} of reference. Good.'
        direction = 'hotter' if diff > 0 else 'below'
        severity  = 'slightly' if abs(diff) < 2 else 'notably' if abs(diff) < 4 else 'significantly'
        return f'{band_name}: User mix is {severity} {direction} than reference by {abs(diff):.1f}{unit}.'

    lines.append(delta_label(user['sub_bass'],  ref['sub_bass'],  'SUB BASS (20-60Hz)'))
    lines.append(delta_label(user['bass'],       ref['bass'],      'BASS (60-120Hz)'))
    lines.append(delta_label(user['low_mid'],    ref['low_mid'],   'LOW MID (120-300Hz)'))
    lines.append(delta_label(user['mid'],        ref['mid'],       'MID (300Hz-2kHz)'))
    lines.append(delta_label(user['high_mid'],   ref['high_mid'],  'HIGH MID (2k-8kHz)'))
    lines.append(delta_label(user['air'],        ref['air'],       'AIR (8k-20kHz)'))

    # Dynamics
    dr_diff = user['dynamic_range'] - ref['dynamic_range']
    if abs(dr_diff) < 1:
        lines.append('DYNAMICS: Dynamic range is well matched to reference.')
    elif dr_diff < 0:
        lines.append(f'DYNAMICS: User dynamic range {user["dynamic_range"]:.1f}dB vs reference {ref["dynamic_range"]:.1f}dB. Mix may be over-compressed.')
    else:
        lines.append(f'DYNAMICS: User dynamic range {user["dynamic_range"]:.1f}dB vs reference {ref["dynamic_range"]:.1f}dB. Mix has more dynamic range than reference.')

    # Stereo width
    w_diff = user['stereo_width'] - ref['stereo_width']
    if abs(w_diff) < 0.05:
        lines.append('STEREO WIDTH: Width is well matched to reference.')
    elif w_diff < 0:
        lines.append(f'STEREO WIDTH: User width {user["stereo_width"]:.2f} vs reference {ref["stereo_width"]:.2f}. Mix is narrower than reference.')
    else:
        lines.append(f'STEREO WIDTH: User width {user["stereo_width"]:.2f} vs reference {ref["stereo_width"]:.2f}. Mix is wider than reference.')

    # Mono compatibility
    if user['mono_compatibility'] < 0.6:
        lines.append(f'MONO COMPATIBILITY: Score {user["mono_compatibility"]:.2f} — potential phase issues detected.')
    elif user['mono_compatibility'] < 0.75:
        lines.append(f'MONO COMPATIBILITY: Score {user["mono_compatibility"]:.2f} — minor phase inconsistencies. Check in mono.')
    else:
        lines.append(f'MONO COMPATIBILITY: Score {user["mono_compatibility"]:.2f} — solid mono compatibility.')

    return '\n'.join(lines)


# ============================================================
# START
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)

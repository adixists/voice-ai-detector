import base64
import librosa
import io

def load_audio_from_base64(b64_string):
    audio_bytes = base64.b64decode(b64_string)
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    return audio

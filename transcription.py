# src/ingestion/transcription.py

import os
import tempfile
import logging
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

logger = logging.getLogger(__name__)

# 1) Initialiser UNE SEULE instance WhisperModel (small, int8)
_whisper_model = WhisperModel(
    model_size_or_path="tiny",
    device="cpu",
    compute_type="int8"
)

def download_audio_from_youtube(url: str, out_dir: str = "data/raw_audio") -> str:
    """
    Télécharge l’audio depuis une vidéo YouTube, le convertit en WAV 16 kHz mono,
    et renvoie le chemin local du fichier WAV.
    """
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "yt_%(id)s.%(ext)s"),
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "prefer_ffmpeg": True,
    }

    logger.info(f"Téléchargement de l’audio depuis YouTube : {url}")
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        wav_path = os.path.join(out_dir, f"yt_{info['id']}.wav")
        logger.info(f"Audio téléchargé et converti → {wav_path}")
        return wav_path

def transcribe_file(audio_path: str, beam_size: int = 5) -> str:
    """
    Transcrit un fichier audio local (WAV ou MP3) avec FastWhisper
    et renvoie le texte complet concaténé.
    """
    logger.info(f"Début de la transcription pour : {audio_path}")
    segments, _ = _whisper_model.transcribe(audio_path, beam_size=beam_size)
    transcript = "\n".join(seg.text for seg in segments)
    logger.info("Transcription terminée")
    return transcript


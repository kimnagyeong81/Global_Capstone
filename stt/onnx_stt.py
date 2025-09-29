# stt/onnx_stt.py
import numpy as np
import onnxruntime
import librosa
import torch
import torchaudio  # 설치가 어려우면 멜은 librosa로 대체 가능(말해줘, 대체코드 줄게)

class QuantizedSTT:
    def __init__(self, model_path: str, samplerate: int = 16000):
        self.session = onnxruntime.InferenceSession(model_path)
        self.samplerate = samplerate
        self.n_mels = 80
        self.hop_length = 160
        self.n_fft = 400
        self.mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.samplerate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        # 데모용 토큰 매핑(실제 모델 토크나이저로 교체 권장)
        self.id2token = {i: chr(i + 44032) for i in range(2000)}

    def _mel(self, y: np.ndarray) -> np.ndarray:
        wav = torch.tensor(y).unsqueeze(0)
        mel = self.mel_fn(wav).numpy()
        return mel

    def transcribe_file(self, audio_path: str) -> str:
        y, _ = librosa.load(audio_path, sr=self.samplerate, mono=True)
        if y.size == 0:
            return ""
        mel = self._mel(y)
        mel_len = np.array([mel.shape[2]], dtype=np.int64)
        logits = self.session.run(None, {"mel": mel, "mel_length": mel_len})[0]
        return self._greedy_decode(logits)

    def _greedy_decode(self, logits: np.ndarray) -> str:
        token_ids = np.argmax(logits, axis=-1)[0]
        text, prev = "", -1
        for tid in token_ids:
            tid = int(tid)
            if tid != prev and tid != 0:
                text += self.id2token.get(tid, "?")
            prev = tid
        return text

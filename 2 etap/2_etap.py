import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
from pydub import silence
import librosa
import librosa.display
from scipy.interpolate import interp1d

# Wczytanie pliku audio
filename = input("Podaj nazwę pliku dźwiękowego: ")  # Nazwa pliku do porównania
filename = filename + ".wav"
rate, data = wavfile.read(filename)

sound = AudioSegment.from_file(filename, format="")

# Wycinanie fragmentu pliku
Cut = 1.4
data = data[:int(rate * Cut)]

# Operacje na próbkach powyżej wartości
mask = (np.abs(data) > 0)
times_positive = np.nonzero(mask)[0] / rate
samples_positive = data[mask]

# Znalezienie maksymalnej wartości amplitudy
max_amplitude = np.max(np.abs(samples_positive))

# Use split_on_silence function from pydub.silence
chunks = silence.split_on_silence(sound, min_silence_len=500, silence_thresh=-30)
words = sum(len(silence.split_on_silence(chunk, min_silence_len=500, silence_thresh=-30)) for chunk in chunks)
duration = len(sound) / 1000
wpm = words / duration * 60

y, sr = librosa.load(filename, sr=12000)
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=350)
avg_pitch = np.mean(pitches[pitches > 0])

# Generowanie charakterystyki z amplitudą
plt.figure(figsize=(8, 4))

plt.annotate(f"Ton głosu w pliku : {avg_pitch:.2f} Hz", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.annotate(f"Liczba słów na minutę: {wpm:.2f}", xy=(0.05, 0.8), xycoords='axes fraction', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Dodanie adnotacji z maksymalną wartością amplitudy
plt.annotate(f"Maksymalna amplituda: {max_amplitude:.2f}", xy=(0.05, 0.7), xycoords='axes fraction', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.plot(times_positive, samples_positive)
plt.title("Charakterystyka z amplitudą")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda")
plt.show()

# Generowanie charakterystyki z częstotliwością fundamentalną
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Interpolacja brakujących wartości częstotliwości fundamentalnej
if len(f0) > 0:
    f0_interp = interp1d(np.where(f0 > 0)[0], f0[f0 > 0], kind='linear', fill_value='extrapolate')
    f0 = f0_interp(np.arange(len(f0)))

# Wykres częstotliwości fundamentalnej
plt.figure(figsize=(8, 4))
plt.plot(f0, color='blue')
plt.title("Częstotliwość fundamentalna")
plt.xlabel("Czas [s]")
plt.ylabel("Częstotliwość [Hz]")
plt.show()

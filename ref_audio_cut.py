import torchaudio

wav, sr = torchaudio.load("Monika_lively.wav")

# Keep first 5 seconds
wav = wav[:, :5 * sr]

torchaudio.save("Monika_ref_5s.wav", wav, sr)

print("Saved Monika_ref_5s.wav")

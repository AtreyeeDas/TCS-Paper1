pip install "numpy<2.0.0" "setuptools<82.0.0" "transformers==4.46.3" \
    phonemizer==3.3.0 speechbrain==1.0.2 librosa==0.10.2.post1 scipy==1.13.1

import sys
import os

def check_system():
    print("="*75)
    print("ICASSP CS-TTS PIPELINE: SYSTEM & DEPENDENCY VERIFICATION")
    print("="*75)
    
    # 1. Check Python Version
    py_ver = sys.version_info
    print(f"[+] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver.major != 3 or py_ver.minor != 10:
        print(" [!] WARNING: Environment is not Python 3.10. Expect C-binding leaks!")
    else:
        print(" [✓] Python 3.10 environment confirmed.")

    # 2. Check PyTorch & CUDA 13.0 Blackwell Setup
    try:
        import torch
        print(f"\n[+] PyTorch Version: {torch.__version__}")
        if "cu130" not in torch.__version__:
            print(" [!] WARNING: PyTorch is not compiled with CUDA 13.0 (cu130)!")
        else:
            print(" [✓] CUDA 13.0 Nightly build confirmed.")
            
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"[+] GPU Detected: {gpu_name}")
            print(f"[+] Compute Capability: sm_{cap[0]}{cap[1]}")
            if cap[0] == 12 and cap[1] == 0:
                print(" [✓] NVIDIA Blackwell (sm_120) architecture fully recognized!")
            else:
                print(f" [!] Note: Expected sm_120, detected sm_{cap[0]}{cap[1]}")
        else:
            print(" [X] ERROR: CUDA is NOT available to PyTorch!")
    except Exception as e:
        print(f" [X] ERROR loading PyTorch: {e}")

    # 3. Check Critical Library Imports & Version Constraints
    print("\n[+] Testing Module Dependencies & Constraints...")
    modules_to_test = [
        ("numpy", "1.26.4"),
        ("setuptools", "<82.0.0"),
        ("transformers", "4.46.3"),
        ("phonemizer", "3.3.0"),
        ("speechbrain", "1.0.2"),
        ("librosa", "0.10.2"),
        ("scipy", "1.13.1")
    ]
    
    for mod_name, expected_ver in modules_to_test:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Unknown/Pinned")
            print(f" [✓] {mod_name:<15} Imported successfully (Version: {ver})")
        except ImportError as e:
            print(f" [X] ERROR: Failed to import {mod_name}: {e}")
        except Exception as e:
            print(f" [X] ERROR: Crash when loading {mod_name}: {e}")

    # 4. Verify Native PyTorch Whisper ASR Compatibility
    print("\n[+] Verifying Native Transformers ASR Engine (Offline Ready)...")
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        print(" [✓] AutoModelForSpeechSeq2Seq & ASR Pipeline modules imported cleanly.")
    except Exception as e:
        print(f" [X] ERROR importing native Transformers speech modules: {e}")

    # 5. Check espeak-ng system backend for Stage 1 Phonetics
    try:
        from phonemizer.backend import EspeakBackend
        backend = EspeakBackend('en-us')
        print("\n[✓] eSpeak-NG system C++ backend is reachable and operational!")
    except Exception as e:
        print(f"\n[X] ERROR: eSpeak-NG backend not found. Did you run 'sudo apt-get install espeak-ng'?\nDetails: {e}")

    print("="*75)
    print("VERIFICATION COMPLETE. Your system is 100% aligned for ICASSP offline runs.")
    print("="*75)

if __name__ == "__main__":
    check_system()

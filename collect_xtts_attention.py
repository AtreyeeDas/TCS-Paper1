import os
import json
import soundfile as sf

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


############################################

MODEL_DIR = "/home/spark2/Models/XTTS-v2"

TRANSCRIPT_FILE = "./MUCS_sliced/test/transcripts.txt"

REFERENCE_WAV = "./Monika_ref_5s.wav"

OUTPUT_ROOT = "./xtts_attention_results"

NUM_UTTS = 20

############################################


config = XttsConfig()
config.load_json(os.path.join(MODEL_DIR, "config.json"))

model = Xtts.init_from_config(config)

model.load_checkpoint(
    config,
    checkpoint_dir=MODEL_DIR,
    eval=True,
)

model.cuda()


####################################################

examples = []

with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        parts = line.strip().split("\t", 1)

        if len(parts) != 2:
            continue

        examples.append(parts)

examples = examples[:NUM_UTTS]

print("Loaded", len(examples), "examples")

####################################################

for idx, (uttid, transcript) in enumerate(examples):

    print()

    print("=" * 70)

    print(idx, uttid)

    print(transcript)

    print("=" * 70)

    save_dir = os.path.join(
        OUTPUT_ROOT,
        f"{idx:03d}_{uttid}"
    )

    os.makedirs(save_dir, exist_ok=True)

    result = model.synthesize(

        text=transcript,

        config=config,

        speaker_wav=REFERENCE_WAV,

        language="en",

        attention_save_dir=save_dir,
    )

    wav = result["wav"]

    sf.write(

        os.path.join(save_dir, "generated.wav"),

        wav,

        24000,

    )

    with open(

        os.path.join(save_dir, "transcript.txt"),

        "w",

        encoding="utf8",

    ) as f:

        f.write(transcript)

    meta = {

        "uttid": uttid,

        "text": transcript,

        "num_samples": len(wav),

    }

    with open(

        os.path.join(save_dir, "metadata.json"),

        "w",

        encoding="utf8",

    ) as f:

        json.dump(meta, f, indent=2)

print()

print("DONE")

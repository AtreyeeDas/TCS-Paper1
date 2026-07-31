import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import os

MODEL_DIR="/home/spark2/Models/XTTS-v2"

config=XttsConfig()
config.load_json(os.path.join(MODEL_DIR,"config.json"))

model=Xtts.init_from_config(config)

model.load_checkpoint(
    config,
    checkpoint_dir=MODEL_DIR,
    eval=True,
)

token_file="./xtts_attention_results/000_103085_w5Jyq3XMbb3WwiKQ_0000/text_tokens.pt"

tokens=torch.load(token_file)

print("Shape:",tokens.shape)

tokenizer=model.tokenizer

ids=tokens.squeeze().tolist()

print("\nTOKEN IDS\n")
print(ids)

print("\nDECODED TOKENS\n")

for i,t in enumerate(ids):

    try:
        piece=tokenizer.decode([t])
    except:
        piece=str(t)

    print(f"{i:03d} : {repr(piece)}")

"""
Model successfully loaded.

Shape: torch.Size([1, 66])

TOKEN IDS

[259, 6293, 6592, 2, 6173, 6411, 6211, 2, 973, 175, 32, 2, 6267, 2, 6310, 2, 6285, 6274, 6345, 6306, 2, 17, 1024, 34, 763, 2, 6448, 6278, 2, 6281, 2, 6491, 6308, 6266, 6377, 2, 182, 31, 26, 1070, 52, 2, 6260, 2, 6298, 2, 248, 166, 50, 2, 803, 51, 1518, 25, 2, 6267, 2, 6344, 6270, 2, 6274, 6289, 6179, 6192, 2, 6261]

DECODED TOKENS

000 : '[en]'
001 : 'लि'
002 : 'बर'
003 : ' '
004 : 'ऑ'
005 : 'फि'
006 : 'स'
007 : ' '
008 : 'imp'
009 : 'res'
010 : 's'
011 : ' '
012 : 'में'
013 : ' '
014 : 'एक'
015 : ' '
016 : 'प्र'
017 : 'स्'
018 : 'तु'
019 : 'ति'
020 : ' '
021 : 'd'
022 : 'oc'
023 : 'u'
024 : 'ment'
025 : ' '
026 : 'बना'
027 : 'ना'
028 : ' '
029 : 'और'
030 : ' '
031 : 'बु'
032 : 'नि'
033 : 'या'
034 : 'दी'
035 : ' '
036 : 'fo'
037 : 'r'
038 : 'm'
039 : 'att'
040 : 'ing'
041 : ' '
042 : 'के'
043 : ' '
044 : 'इस'
045 : ' '
046 : 'sp'
047 : 'ok'
048 : 'en'
049 : ' '
050 : 'tu'
051 : 'to'
052 : 'ria'
053 : 'l'
054 : ' '
055 : 'में'
056 : ' '
057 : 'आप'
058 : 'का'
059 : ' '
060 : 'स्'
061 : 'वा'
062 : 'ग'
063 : 'त'
064 : ' '
065 : 'है'
"""

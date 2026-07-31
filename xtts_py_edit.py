gpt_codes, generation_outputs = self.gpt.generate(
    cond_latents=gpt_cond_latent,
    text_inputs=text_tokens,
    input_tokens=None,
    do_sample=do_sample,
    top_p=top_p,
    top_k=top_k,
    temperature=temperature,
    num_return_sequences=self.gpt_batch_size,
    num_beams=num_beams,
    length_penalty=length_penalty,
    repetition_penalty=repetition_penalty,

    output_attentions=True,
    output_hidden_states=True,
    output_scores=True,
    return_dict_in_generate=True,

    **hf_generate_kwargs,
)
print("\n==========================")
print("GENERATION OUTPUT")
print("==========================")

print(type(generation_outputs))

print()

print(generation_outputs.keys())

print()

print("Sequences shape:",
      generation_outputs.sequences.shape)

print()

if generation_outputs.attentions is None:
    print("ATTENTIONS = None")
else:

    print("Attention timesteps:",
          len(generation_outputs.attentions))

    print()

    first = generation_outputs.attentions[0]

    print("Type of first timestep:", type(first))

    print("Number of layers:",
          len(first))

    print()

    print("Layer0 shape:",
          first[0].shape)


from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("/home/spark2/Models/XTTS-v2/config.json")

model = Xtts.init_from_config(config)

model.load_checkpoint(
    config,
    checkpoint_dir="/home/spark2/Models/XTTS-v2",
    eval=True,
)

model.cuda()

result = model.synthesize(

    text="Hello namaste kaise ho",

    config=config,

    speaker_wav="Monika_ref_5s.wav",

    language="en",
)

"""
Output:
==========================
GENERATION OUTPUT
==========================
<class 'transformers.generation.utils.GenerateDecoderOnlyOutput'>

odict_keys(['sequences', 'scores', 'attentions', 'hidden_states', 'past_key_values'])

Sequences shape: torch.Size([1, 89])

Attention timesteps: 41

Type of first timestep: <class 'tuple'>
Number of layers: 30

Layer0 shape: torch.Size([1, 16, 48, 48])
"""

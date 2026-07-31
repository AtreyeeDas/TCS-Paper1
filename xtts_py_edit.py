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
    return_dict_in_generate=True,
    output_scores=True,

    **hf_generate_kwargs,
)


attentions = generation_outputs.attentions

print(type(attentions))

print(len(attentions))



type(generation_outputs)

generation_outputs.keys()

type(generation_outputs.attentions)

len(generation_outputs.attentions)


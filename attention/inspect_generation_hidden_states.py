import pickle

FILE = "./xtts_attention_results/000_103085_w5Jyq3XMbb3WwiKQ_0000/generation_hidden_states.pkl"

with open(FILE, "rb") as f:
    hs = pickle.load(f)

print(type(hs))

print("Timesteps:", len(hs))

print()

print("First timestep type:", type(hs[0]))

print("Layers:", len(hs[0]))

print()

for i, x in enumerate(hs[0]):
    print(i, x.shape)


import pickle

FILE="./xtts_attention_results/000_103085_w5Jyq3XMbb3WwiKQ_0000/generation_attentions.pkl"

with open(FILE,"rb") as f:
    att=pickle.load(f)

print(type(att[0]))

print(len(att[0]))

for i,a in enumerate(att[0]):
    print(i,a.shape)

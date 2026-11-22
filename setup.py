import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
MASK_ID = 50257

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForMaskedLM.from_pretrained(
    "./mdlm-local", trust_remote_code=True
).to(device).eval()

# A sentence with two words replaced by [MASK]
x = tok("The capital of France is Paris and it is a big city.", return_tensors="pt").input_ids
print(x)
x[0, 1] = MASK_ID   # "is"
x[0, 3] = MASK_ID   # "is"
x[0, 4] = MASK_ID   # "is"
x[0, 5] = MASK_ID   # "Paris"
x = x.to(device)

with torch.no_grad():
    out = model(input_ids=x, timesteps=torch.zeros(1, device=device))
logits = out.logits if hasattr(out, "logits") else out
logits[..., MASK_ID] = -float("inf")

conf, guess = logits.softmax(-1).max(-1)
for pos in [1, 2, 3, 4, 5]:
    print(pos, repr(tok.decode(guess[0, pos])), round(conf[0, pos].item(), 3))
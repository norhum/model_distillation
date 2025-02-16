from hellaswag import iterate_examples, render_example, get_most_likely_row
import torch
from transformers import AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model =  AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-2.7B",
            device_map="auto",
            torch_dtype=torch.float16  # Use float16 for teacher
        )

num_correct_norm = 0
num_total = 0
model.eval() 
for i, example in enumerate(iterate_examples("val")):
    # render the example into tokens and labels
    _, tokens, mask, label = render_example(example)
    tokens = tokens.to(device)
    mask = mask.to(device)
    # get the logits
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(tokens).logits
        pred_norm = get_most_likely_row(tokens, mask, logits)
    num_total += 1
    num_correct_norm += int(pred_norm == label)

# reduce the stats across all processes
num_total = int(num_total)  # Just ensure it's an integer
num_correct_norm = int(num_correct_norm)  # Convert to int if needed

acc_norm = num_correct_norm / num_total

print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
print()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from model import GPT

model_name = "EleutherAI/gpt-neo-2.7B"  # Or GPT-NeoX-20B / Pythia-12B
tokenizer = AutoTokenizer.from_pretrained(model_name)
teacher_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Example input
text = "The future of AI is"
inputs = tokenizer(text, return_tensors="pt").to(device)

# Get logits
with torch.no_grad():
    t_outputs = teacher_model(**inputs)
    t_logits = t_outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

student_model = GPT.from_pretrained("gpt2")
student_model.to(device)

with torch.no_grad():
    s_outputs = student_model(inputs["input_ids"])
    s_logits = s_outputs[0] # Shape: [batch_size, seq_len, vocab_size]

def softmax_with_temperature(logits, temperature):
    return torch.nn.functional.softmax(logits / temperature, dim=-1)

def distillation_loss(student_logits, teacher_logits, temperature):
    # Softened teacher probabilities
    teacher_probs = softmax_with_temperature(teacher_logits, temperature)
    
    # Softened student logits
    student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    
    # Calculate KL divergence
    kl_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    return kl_loss

def total_loss(student_logits, teacher_logits, student_labels, temperature, alpha, criterion):
    # Distillation loss
    dist_loss = distillation_loss(student_logits, teacher_logits, temperature)
    
    # Cross-entropy loss
    ce_loss = criterion(student_logits.view(-1, student_logits.size(-1)), student_labels.view(-1))
    
    # Total loss
    total = alpha * dist_loss + (1 - alpha) * ce_loss
    return total

# Example training loop
optimizer.zero_grad()

# Forward pass
student_logits = student_model(input_ids)
teacher_logits = teacher_model(input_ids)  # Teacher is usually fixed

# Compute loss
loss = total_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5, criterion=criterion)

# Backpropagate and update
loss.backward()
optimizer.step()

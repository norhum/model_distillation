import torch
from transformers import GPT2Tokenizer
from model import GPT
from tqdm import tqdm

def compare_models(original_model_name="gpt2", distilled_model_path="path/to/distilled/model"):
    """
    Compare original GPT-2 with its distilled version
    """
    # Load models and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(original_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading original model...")
    original_model = GPT.from_pretrained(original_model_name)
    
    print("Loading distilled model...")
    distilled_model = GPT.from_pretrained(original_model_name)  # Initialize a fresh model with the correct config
    distilled_model.load_state_dict(torch.load(distilled_model_path))  # Load weights
    distilled_model.eval()  # Set to evaluation mode if needed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_model.to(device)
    distilled_model.to(device)
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "Climate change will affect",
        "The most important scientific discovery",
        "In the next ten years, technology will",
        "The relationship between humans and machines",
    ]
    
    results = {
        "speed_comparison": [],
        "token_comparison": [],
        "memory_usage": {},
        "generated_texts": []
    }
    
    # Compare generation quality
    print("\nComparing generation quality...")
    for prompt in tqdm(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            original_output = original_model.generate(
                inputs["input_ids"],
                max_length=30,
                num_return_sequences=1,
            )

        with torch.no_grad():
            distilled_output = distilled_model.generate(
                inputs["input_ids"],
                max_length=30,
                num_return_sequences=1,
            )

        # Decode outputs
        original_output = original_output.squeeze(0)  # Remove batch dim -> shape (1, 106)
        original_text = tokenizer.decode(original_output[0].tolist(), skip_special_tokens=True)
        distilled_output = distilled_output.squeeze(0) 
        distilled_text = tokenizer.decode(distilled_output[0].tolist(), skip_special_tokens=True)
        
        results["generated_texts"].append({
            "prompt": prompt,
            "original": original_text,
            "distilled": distilled_text
        })
    
    # Compare model sizes
    def get_model_size(model):
        return sum(p.numel() for p in model.parameters()) / 1e6  # Size in millions of parameters
    
    results["memory_usage"]["original"] = get_model_size(original_model)
    
    print(f"Model: {results['memory_usage']['original']:.2f}M parameters")

    print("\nGeneration Examples:")
    for gen in results["generated_texts"]:
        print(f"\nPrompt: {gen['prompt']}")
        print(f"Original: {gen['original']}")
        print(f"Distilled: {gen['distilled']}")
        print("-" * 80)
    
    return results

if __name__ == "__main__":
    # Replace with your distilled model path
    results = compare_models(
        original_model_name="gpt2",
        distilled_model_path="distilled_model\student_model.pth"
    )

    # hellaswag score
    import matplotlib.pyplot as plt
    import re  

    with open(r"..\logs\hellaswag", "r") as f:
        output = [line.strip() for line in f.readlines()]

    # Data extraction using regular expressions
    x_values = []
    y_values = []

    for line in output:  # Now we iterate through the *list* of lines
        match = re.match(r'(\d+)\s+\w+\s+([\d.]+)', line)
        if match:
            x_values.append(int(match.group(1)))  # Index (convert to integer)
            y_values.append(float(match.group(2)))  # Score (convert to float)

    plt.plot(x_values, y_values, marker='o', linestyle='-', label='HellaSwag Score') 

    target_value = 0.2955  
    plt.axhline(y=target_value, color='r', linestyle='--', label=f'Target: {target_value}')

    # Customize the plot
    plt.xlabel('Index')
    plt.ylabel('Score')
    plt.title('HellaSwag Score over Time')
    plt.grid(False)  
    plt.legend()   

    # Show the plot
    plt.show()

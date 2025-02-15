import torch
from transformers import GPT2Tokenizer
from model import GPT
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Compare generation speed and quality
    print("\nComparing generation speed and quality...")
    for prompt in tqdm(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Time original model
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            original_output = original_model.generate(
                inputs["input_ids"],
                max_length=30,
                num_return_sequences=1,
            )
        torch.cuda.synchronize()
        original_time = time.time() - start_time
        
        # Time distilled model
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            distilled_output = distilled_model.generate(
                inputs["input_ids"],
                max_length=30,
                num_return_sequences=1,
            )
        torch.cuda.synchronize()
        distilled_time = time.time() - start_time

        # Decode outputs
        original_output = original_output.squeeze(0)  # Remove batch dim -> shape (1, 106)
        original_text = tokenizer.decode(original_output[0].tolist(), skip_special_tokens=True)
        distilled_output = distilled_output.squeeze(0) 
        distilled_text = tokenizer.decode(distilled_output[0].tolist(), skip_special_tokens=True)
        
        results["speed_comparison"].append({
            "prompt": prompt,
            "original_time": original_time,
            "distilled_time": distilled_time,
            "speedup": original_time / distilled_time
        })
        
        results["generated_texts"].append({
            "prompt": prompt,
            "original": original_text,
            "distilled": distilled_text
        })
    
    # Compare model sizes
    def get_model_size(model):
        return sum(p.numel() for p in model.parameters()) / 1e6  # Size in millions of parameters
    
    results["memory_usage"]["original"] = get_model_size(original_model)
    results["memory_usage"]["distilled"] = get_model_size(distilled_model)
    
    # Print results
    print("\n=== Model Comparison Results ===")
    print("\nModel Sizes:")
    print(f"Original Model: {results['memory_usage']['original']:.2f}M parameters")
    print(f"Distilled Model: {results['memory_usage']['distilled']:.2f}M parameters")
    print(f"Size Reduction: {(1 - results['memory_usage']['distilled']/results['memory_usage']['original'])*100:.1f}%")
    
    avg_speedup = np.mean([r["speedup"] for r in results["speed_comparison"]])
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    
    print("\nGeneration Examples:")
    for gen in results["generated_texts"]:
        print(f"\nPrompt: {gen['prompt']}")
        print(f"Original: {gen['original']}")
        print(f"Distilled: {gen['distilled']}")
        print("-" * 80)
    
    # Create visualization
    plt.figure(figsize=(10, 5))
    speedups = [r["speedup"] for r in results["speed_comparison"]]
    plt.bar(range(len(speedups)), speedups)
    plt.axhline(y=1, color='r', linestyle='--', label='Original Speed')
    plt.xlabel("Test Cases")
    plt.ylabel("Speedup Factor")
    plt.title("Distilled Model Speedup")
    plt.legend()
    plt.savefig('speedup_comparison.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    # Replace with your distilled model path
    results = compare_models(
        original_model_name="gpt2",
        distilled_model_path="distilled_model\student_model.pth"
    )

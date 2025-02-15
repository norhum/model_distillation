import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from hellaswag import iterate_examples, render_example, get_most_likely_row
from tqdm import tqdm
import wandb
from model import GPT  
import os
import json
from datetime import datetime

class DistillationTrainer:
    def __init__(
        self,
        teacher_model_name: str,
        student_model_name: str,
        temperature: float = 2.0,
        alpha: float = 0.5,
        learning_rate: float = 1e-4,
        max_length: int = 128,
        batch_size: int = 8
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher_model_name = teacher_model_name
        self.temperature = temperature
        self.alpha = alpha
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize teacher model
        print("Loading teacher model...")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.teacher_model.eval()  # Teacher model should always be in eval mode
        
        # Initialize student model
        print("Loading student model...")
        self.student_model = GPT.from_pretrained(student_model_name)
        self.student_model.to(self.device)
        
        # Initialize optimizer and criterion
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()

    def prepare_batch(self, texts):
        # Tokenize and prepare input batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Create labels (shift input_ids left by 1)
        labels = input_ids.clone()
        labels = torch.roll(labels, -1, dims=1)
        labels[:, -1] = -100  # Ignore last token prediction
        
        return input_ids, attention_mask, labels

    @staticmethod
    def softmax_with_temperature(logits, temperature):
        return torch.nn.functional.softmax(logits / temperature, dim=-1)

    def distillation_loss(self, student_logits, teacher_logits):
        """Calculate the distillation loss using KL divergence"""
        teacher_probs = self.softmax_with_temperature(teacher_logits, self.temperature)
        student_log_probs = torch.nn.functional.log_softmax(
            student_logits / self.temperature,
            dim=-1
        )
        
        # Calculate KL divergence loss
        kl_loss = torch.nn.functional.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        )
        
        return kl_loss * (self.temperature ** 2)  # Scale loss by temperature squared

    def train_step(self, input_ids, attention_mask, labels):
        self.optimizer.zero_grad()
        
        # Get teacher logits (with no gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits
        
        # Get student logits
        student_outputs = self.student_model(
            input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_outputs["logits"]
        
        # Calculate losses
        dist_loss = self.distillation_loss(student_logits, teacher_logits)
        ce_loss = self.criterion(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        # Combine losses
        total_loss = (self.alpha * dist_loss) + ((1 - self.alpha) * ce_loss)
        
        # Backpropagate and update
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'distillation_loss': dist_loss.item(),
            'ce_loss': ce_loss.item()
        }
    
    def save_model(self, output_dir="./distilled_model"):
        """Save the distilled model and tokenizer"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving model to {output_dir}")
        torch.save(self.student_model.state_dict(), os.path.join(output_dir, "student_model.pth"))
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config file with distillation parameters
        config = {
            "teacher_model": self.teacher_model_name,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "max_length": self.max_length,
            "distillation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(output_dir, "distillation_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        return output_dir

    def train(self, train_texts, eval_texts, num_epochs=3):
        # Initialize wandb for tracking
        wandb.init(project="model-distillation")
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_texts,
            batch_size=self.batch_size,
            shuffle=True
        )
        eval_dataloader = DataLoader(
            eval_texts,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Create learning rate scheduler
        num_training_steps = len(train_dataloader) * num_epochs
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        for epoch in range(num_epochs):
            self.student_model.train()
            total_train_loss = 0
            
            # Training
            train_progress_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}')
            for batch_texts in train_progress_bar:
                input_ids, attention_mask, labels = self.prepare_batch(batch_texts)
                step_losses = self.train_step(input_ids, attention_mask, labels)
                
                total_train_loss += step_losses['total_loss']
                lr_scheduler.step()
                
                # Update progress bar
                train_progress_bar.set_postfix({
                    'total_loss': step_losses['total_loss'],
                    'dist_loss': step_losses['distillation_loss'],
                    'ce_loss': step_losses['ce_loss']
                })
                
                # Log to wandb
                wandb.log(step_losses)
            
            # Evaluation
            self.student_model.eval()
            total_eval_loss = 0
            
            with torch.no_grad():
                eval_progress_bar = tqdm(eval_dataloader, desc=f'Evaluating Epoch {epoch + 1}')
                for batch_texts in eval_progress_bar:
                    input_ids, attention_mask, labels = self.prepare_batch(batch_texts)
                    
                    student_outputs = self.student_model(
                        input_ids,
                        attention_mask=attention_mask
                    )
                    student_logits = student_outputs["logits"]
                    
                    eval_loss = self.criterion(
                        student_logits.view(-1, student_logits.size(-1)),
                        labels.view(-1)
                    )
                    total_eval_loss += eval_loss.item()
            
            # Log epoch metrics
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            
            wandb.log({
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'avg_eval_loss': avg_eval_loss
            })

            # for every epoch, evaluate hellaswag and log the data instead of returning it at the end. 
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(self.device)
                mask = mask.to(self.device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        logits = self.student_model(tokens)["logits"]
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)

            # reduce the stats across all processes
            num_total = int(num_total)  # Just ensure it's an integer
            num_correct_norm = int(num_correct_norm)  # Convert to int if needed

            acc_norm = num_correct_norm / num_total
    
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            print()
            with open(r"logs/hellaswag", "a") as f:
                f.write(f"{epoch} hellaswag {acc_norm:.4f}\n")
            
            print(f'Epoch {epoch + 1}:')
            print(f'Average Training Loss: {avg_train_loss:.4f}')
            print(f'Average Evaluation Loss: {avg_eval_loss:.4f}')
            
        wandb.finish()
        # Save model after training
        self.save_model("./distilled_model")

# Example usage
if __name__ == "__main__":
    trainer = DistillationTrainer(
        teacher_model_name="EleutherAI/gpt-neo-2.7B",
        student_model_name="gpt2",
        temperature=2.0,
        alpha=0.5,
        learning_rate=1e-4,
        max_length=128,
        batch_size=8
    )
    
    # Example training data (replace with your actual dataset)
    train_texts = [
        "The future of AI is",
        "Machine learning can be used to",
        # ... more training texts
    ]
    
    eval_texts = [
        "Deep learning has revolutionized",
        "Neural networks are capable of",
        # ... more evaluation texts
    ]
    
    trainer.train(train_texts, eval_texts, num_epochs=3)

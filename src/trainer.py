import torch
import torch.nn as nn
import torch.nn.functional as F  
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
        self.student_model_name = student_model_name  
        self.temperature = temperature
        self.alpha = alpha
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate 

        # Initialize tokenizer (shared)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize teacher model
        print("Loading teacher model...")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            device_map="auto",
            torch_dtype=torch.float16  # Use float16 for teacher
        )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False  

        # Initialize student model
        print("Loading student model...")
        self.student_model = GPT.from_pretrained(student_model_name) 
        self.student_model.to(self.device)

        # Optimizer and criterion
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()  


    def prepare_batch(self, batch):
        """Prepare a batch for training/evaluation."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device)  # Correct/incorrect labels (0 or 1)
        return input_ids, attention_mask, labels


    def train_step(self, input_ids, attention_mask, labels):
        """Perform a single training step."""
        self.optimizer.zero_grad()

        # Teacher forward pass (no gradient calculation)
        with torch.no_grad():
            teacher_logits = self.teacher_model(input_ids, attention_mask=attention_mask).logits

        # Student forward pass
        student_outputs = self.student_model(input_ids, attention_mask=attention_mask)
        student_logits = student_outputs["logits"]

        # --- Distillation Loss ---
        # 1. Softmax with temperature (for both teacher and student)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)

        # 2. KL Divergence Loss
        distillation_loss = F.kl_div(torch.log(student_probs), teacher_probs, reduction='batchmean') * (self.temperature ** 2)

        # 3. Cross-Entropy Loss on *hard labels* (for student)
        #    Reshape logits and labels for CrossEntropyLoss
        reshaped_student_logits = student_logits[:, -1, :].view(-1, student_logits.size(-1)) # Shape: (batch_size, num_choices, vocab_size) -> (batch_size * num_choices, vocab_size)
        reshaped_labels = labels.view(-1)  #Shape: (batch_size, num_choices) -> (batch_size * num_choices)
        ce_loss = self.criterion(reshaped_student_logits, reshaped_labels)


        # 4. Combine losses
        loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss

        loss.backward()
        self.optimizer.step()

        return {
            'total_loss': loss.item(),
            'distillation_loss': distillation_loss.item(),
            'ce_loss': ce_loss.item()
        }

    def save_model(self, output_dir="./distilled_model"):
        """Save the distilled model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving model to {output_dir}")
        # Save only the student model's state dict
        torch.save(self.student_model.state_dict(), os.path.join(output_dir, "student_model.pth"))
        self.tokenizer.save_pretrained(output_dir)

        # Save config file with distillation parameters
        config = {
            "teacher_model": self.teacher_model_name,
            "student_model": self.student_model_name,  
            "temperature": self.temperature,
            "alpha": self.alpha,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,  
            "distillation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(output_dir, "distillation_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        return output_dir
    def evaluate(self, val_dataloader):
        """Evaluate the model on the validation set."""
        self.student_model.eval()
        total_eval_accuracy = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.student_model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]

                # Get predictions (highest logit)
                # predictions = torch.argmax(logits, dim=-1) #No need for this

                #Reshape the logits
                reshaped_logits = logits[:, -1, :]

                #Since the batch size now reflect number of choices * number of questions, we reshape the predictions to (num_questions, num_choices)
                reshaped_predictions = reshaped_logits.argmax(dim=-1).view(-1, 4)
                labels = labels.view(-1,4)

                #Compare the predicted choice index with the true label index
                correct_predictions = (reshaped_predictions == labels.argmax(dim=-1, keepdim=True)).sum().item()
                total_eval_accuracy += correct_predictions / reshaped_predictions.size(0) #correct predictions/ number of questions


        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print(f"Validation Accuracy: {avg_val_accuracy:.4f}")
        return avg_val_accuracy

    def train(self, train_dataset, val_dataset, num_epochs=3):
        """Train the student model using distillation."""

        # DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Scheduler (after optimizer initialization)
        num_training_steps = len(train_dataloader) * num_epochs
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # Initialize wandb
        wandb.init(project="model-distillation", config={
            "teacher_model": self.teacher_model_name,
            "student_model": self.student_model_name,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "num_epochs": num_epochs
        })

        for epoch in range(num_epochs):
            self.student_model.train()
            total_train_loss = 0

            train_progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            for batch in train_progress_bar:
                input_ids, attention_mask, labels = self.prepare_batch(batch)
                step_losses = self.train_step(input_ids, attention_mask, labels)
                total_train_loss += step_losses['total_loss']

                lr_scheduler.step()

                train_progress_bar.set_postfix({
                    'total_loss': step_losses['total_loss'],
                    'dist_loss': step_losses['distillation_loss'],
                    'ce_loss': step_losses['ce_loss']
                })

                wandb.log(step_losses)  # Log individual step losses

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss:.4f}")

            # --- Evaluation ---
            avg_val_accuracy = self.evaluate(val_dataloader)

            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_accuracy": avg_val_accuracy  
            })
            
            # for every epoch, evaluate hellaswag and log the data instead of returning it at the end.
            num_correct_norm = 0
            num_total = 0
            self.student_model.eval() 
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
            # Ensure the "logs" directory exists
            os.makedirs("logs", exist_ok=True)
            with open(r"logs/hellaswag", "a") as f:
                f.write(f"{epoch} hellaswag {acc_norm:.4f}\n")
            wandb.log({"hellaswag_accuracy": acc_norm})

        wandb.finish()
        self.save_model() 

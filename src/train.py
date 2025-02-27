# train.py
import datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
import wandb
import os
import json
from datetime import datetime
from model import GPT  
from hellaswag import iterate_examples, render_example, get_most_likely_row, load_hellaswag

class OpenBookQADataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for item in dataset:
            question = item['question_stem']
            choices = item['choices']
            correct_answer_index = ord(item['answerKey']) - ord('A')

            if correct_answer_index < 0 or correct_answer_index >= len(choices['text']):
                print(f"Warning: Invalid answer key '{item['answerKey']}'. Skipping example.")
                continue

            for idx, choice_text in enumerate(choices['text']):
                text = f"Question: {question} Choice: {choice_text}"
                encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
                label = torch.tensor(1 if idx == correct_answer_index else 0)
                self.data.append({
                    'input_ids': encoded['input_ids'].squeeze(),
                    'attention_mask': encoded['attention_mask'].squeeze(),
                    'label': label
                })
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def prepare_openbookqa_data(tokenizer, max_length=128):
    print("Loading OpenBookQA dataset...")
    dataset = datasets.load_dataset("openbookqa", "main")
    train_dataset = OpenBookQADataset(dataset['train'], tokenizer, max_length)
    val_dataset = OpenBookQADataset(dataset['validation'], tokenizer, max_length)
    return train_dataset, val_dataset

class DistillationTrainer:
    def __init__(
        self,
        teacher_model_name: str,
        student_model_name: str,
        temperature: float = 2.0,
        alpha_start: float = 0.7,
        alpha_end: float = 0.3,
        learning_rate: float = 5e-6,
        max_length: int = 128,
        batch_size: int = 8,
        freeze_layers: bool = True,
        hellaswag_weight: float = 0.8,
        curriculum_epochs: int = 3
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.temperature = temperature
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha = alpha_start
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.freeze_layers = freeze_layers
        self.hellaswag_weight = hellaswag_weight
        self.curriculum_epochs = curriculum_epochs

        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading teacher model...")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        print("Loading student model...")
        self.student_model = GPT.from_pretrained(student_model_name)
        self.student_model.to(self.device)

        if self.freeze_layers:
            for i in range(6):
                for param in self.student_model.transformer.h[i].parameters():
                    param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student_model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()

    def prepare_batch(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device)
        return input_ids, attention_mask, labels

    def train_step(self, input_ids, attention_mask, labels):
        self.optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = self.teacher_model(input_ids, attention_mask=attention_mask).logits

        student_outputs = self.student_model(input_ids, attention_mask=attention_mask)
        student_logits = student_outputs["logits"]

        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)

        distillation_loss = F.kl_div(torch.log(student_probs), teacher_probs, reduction='batchmean') * (self.temperature ** 2)

        reshaped_student_logits = student_logits[:, -1, :].view(-1, student_logits.size(-1))
        reshaped_labels = labels.view(-1)
        ce_loss = self.criterion(reshaped_student_logits, reshaped_labels)
        loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss

        loss.backward()
        self.optimizer.step()

        return {'total_loss': loss.item(), 'distillation_loss': distillation_loss.item(), 'ce_loss': ce_loss.item()}

    def save_model(self, output_dir="./distilled_model"):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model to {output_dir}")
        torch.save(self.student_model.state_dict(), os.path.join(output_dir, "student_model.pth"))
        self.tokenizer.save_pretrained(output_dir)
        config = {
            "teacher_model": self.teacher_model_name,
            "student_model": self.student_model_name,
            "temperature": self.temperature,
            "alpha_start": self.alpha_start,
            "alpha_end": self.alpha_end,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "freeze_layers": self.freeze_layers,
            "hellaswag_weight": self.hellaswag_weight,
            "curriculum_epochs": self.curriculum_epochs,
            "distillation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(output_dir, "distillation_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        return output_dir

    def evaluate(self, val_dataloader):
        self.student_model.eval()
        total_eval_accuracy = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                input_ids, attention_mask, labels = self.prepare_batch(batch)
                outputs = self.student_model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                reshaped_logits = logits[:, -1, :]
                reshaped_predictions = reshaped_logits.argmax(dim=-1).view(-1, 4)
                labels = labels.view(-1,4)
                correct_predictions = (reshaped_predictions == labels.argmax(dim=-1, keepdim=True)).sum().item()
                total_eval_accuracy += correct_predictions / reshaped_predictions.size(0)

        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print(f"Validation Accuracy: {avg_val_accuracy:.4f}")
        return avg_val_accuracy

    def train(self, train_dataset, val_dataset, hellaswag_train_dataset, num_epochs=3):
        # Create Subsets *before* ConcatDataset
        hellaswag_subset = Subset(hellaswag_train_dataset, torch.randperm(len(hellaswag_train_dataset))[:int(len(hellaswag_train_dataset) * self.hellaswag_weight)])
        combined_dataset = ConcatDataset([train_dataset, hellaswag_subset])

        train_dataloader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        num_training_steps = len(train_dataloader) * num_epochs
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        wandb.init(project="model-distillation", config={
            "teacher_model": self.teacher_model_name,
            "student_model": self.student_model_name,
            "temperature": self.temperature,
            "alpha_start": self.alpha_start,
            "alpha_end": self.alpha_end,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "freeze_layers": self.freeze_layers,
            "hellaswag_weight": self.hellaswag_weight,
            "curriculum_epochs": self.curriculum_epochs,
            "num_epochs": num_epochs,
        })

        for epoch in range(num_epochs):
            self.student_model.train()
            total_train_loss = 0

            if epoch < self.curriculum_epochs:
                self.alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * (epoch / self.curriculum_epochs)
                hellaswag_subset_size = int(len(hellaswag_train_dataset) * self.hellaswag_weight * (1 - epoch / self.curriculum_epochs))

                # Create new subset INDICES
                hellaswag_subset_indices = torch.randperm(len(hellaswag_train_dataset))[:hellaswag_subset_size]
                hellaswag_subset = Subset(hellaswag_train_dataset, hellaswag_subset_indices)
                combined_dataset = ConcatDataset([train_dataset, hellaswag_subset])  # Use new subset
                train_dataloader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True)  # New DataLoader

            train_progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            for batch in train_progress_bar:
                input_ids, attention_mask, labels = self.prepare_batch(batch)
                step_losses = self.train_step(input_ids, attention_mask, labels)
                total_train_loss += step_losses['total_loss']
                lr_scheduler.step()

                train_progress_bar.set_postfix({
                    'total_loss': step_losses['total_loss'],
                    'dist_loss': step_losses['distillation_loss'],
                    'ce_loss': step_losses['ce_loss'],
                    'alpha': self.alpha
                })
                wandb.log(step_losses)

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss:.4f}")
            avg_val_accuracy = self.evaluate(val_dataloader)

            wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_train_loss, "avg_val_accuracy": avg_val_accuracy, "alpha": self.alpha})

            num_correct_norm = 0
            num_total = 0
            self.student_model.eval()
            for i, example in enumerate(iterate_examples("val")):
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
            os.makedirs("logs", exist_ok=True)
            with open(r"logs/hellaswag", "a") as f:
                f.write(f"{epoch} hellaswag {acc_norm:.4f}\n")
            wandb.log({"hellaswag_accuracy": acc_norm})

        wandb.finish()
        self.save_model()

if __name__ == "__main__":
    trainer = DistillationTrainer(
        teacher_model_name="EleutherAI/gpt-neo-2.7B",
        student_model_name="gpt2",
        temperature=1.5,
        alpha_start=0.7,
        alpha_end=0.3,
        learning_rate=5e-6,
        max_length=128,
        batch_size=8,
        freeze_layers=True,
        hellaswag_weight=0.2,
        curriculum_epochs=3
    )

    train_dataset, val_dataset = prepare_openbookqa_data(trainer.tokenizer, max_length=trainer.max_length)
    hellaswag_train_dataset = load_hellaswag("train", trainer.tokenizer, max_seq_length=trainer.max_length)
    trainer.train(train_dataset, val_dataset, hellaswag_train_dataset, num_epochs=10)
    trainer.save_model("distilled_model")
    print("Training complete. Model saved.")

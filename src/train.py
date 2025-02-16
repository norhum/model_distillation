import datasets
from torch.utils.data import Dataset
import torch

class OpenBookQADataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for item in dataset:
            question = item['question_stem']
            choices = item['choices']
            correct_answer_index = ord(item['answerKey']) - ord('A')  # Convert 'A', 'B', 'C', 'D' to 0, 1, 2, 3

            # Check if the answer key is valid
            if correct_answer_index < 0 or correct_answer_index >= len(choices['text']):
                print(f"Warning: Invalid answer key '{item['answerKey']}'. Skipping example.")
                continue

            for idx, choice_text in enumerate(choices['text']):
                # Format: "Question: [question text] Choice: [choice text]"
                text = f"Question: {question} Choice: {choice_text}"
                encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')

                # Create a label tensor: 1 if correct choice, 0 otherwise
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
    """
    Prepare OpenBookQA dataset for training and validation.
    """
    print("Loading OpenBookQA dataset...")
    dataset = datasets.load_dataset("openbookqa", "main")

    train_dataset = OpenBookQADataset(dataset['train'], tokenizer, max_length)
    val_dataset = OpenBookQADataset(dataset['validation'], tokenizer, max_length)
    # test_dataset = OpenBookQADataset(dataset['test'], tokenizer, max_length) #If you have test set

    return train_dataset, val_dataset

if __name__ == "__main__":
    from trainer import DistillationTrainer  

    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model_name="EleutherAI/gpt-neo-2.7B", 
        student_model_name="gpt2",
        temperature=1.5,
        alpha=0.7,
        learning_rate=5e-5,  
        max_length=128,
        batch_size=8 
    )

    # Prepare data:  Use OpenBookQA
    train_dataset, val_dataset = prepare_openbookqa_data(trainer.tokenizer, max_length=trainer.max_length)

    # Train the model
    trainer.train(train_dataset, val_dataset, num_epochs=3)
    trainer.save_model("distilled_model.pt") 
    print("Training complete. Model saved.")

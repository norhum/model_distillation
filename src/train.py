import datasets
from torch.utils.data import Dataset
import random
import torch

class TextDataset(Dataset):
    def __init__(self, texts, chunk_size=128):
        self.texts = texts
        self.chunk_size = chunk_size
        
        # Process texts into chunks
        self.chunks = []
        for text in texts:
            # Split into chunks of approximately chunk_size characters
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            # Only keep chunks that are long enough
            chunks = [chunk for chunk in chunks if len(chunk) >= chunk_size // 2]
            self.chunks.extend(chunks)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]

def prepare_wikitext_data(split_ratio=0.9):
    """
    Prepare WikiText-2 dataset for training and validation
    """
    # Load WikiText-2 dataset
    print("Loading WikiText-2 dataset...")
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Combine all texts from the training set
    train_texts = dataset['train']['text']
    
    # Filter out empty lines and very short texts
    train_texts = [text for text in train_texts if len(text.strip()) > 100]
    
    # Shuffle the texts
    random.seed(42)
    random.shuffle(train_texts)
    
    # Split into train and validation
    split_idx = int(len(train_texts) * split_ratio)
    train_data = train_texts[:split_idx]
    val_data = train_texts[split_idx:]
    
    print(f"Number of training texts: {len(train_data)}")
    print(f"Number of validation texts: {len(val_data)}")
    
    return train_data, val_data

def prepare_custom_data(file_path, split_ratio=0.9):
    """
    Prepare custom dataset from a text file
    Each line should be a separate text sample
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    # Filter out empty lines and very short texts
    texts = [text.strip() for text in texts if len(text.strip()) > 100]
    
    # Shuffle the texts
    random.seed(42)
    random.shuffle(texts)
    
    # Split into train and validation
    split_idx = int(len(texts) * split_ratio)
    train_data = texts[:split_idx]
    val_data = texts[split_idx:]
    
    print(f"Number of training texts: {len(train_data)}")
    print(f"Number of validation texts: {len(val_data)}")
    
    return train_data, val_data

def get_sample_texts():
    """
    Get a small sample of texts for testing
    """
    return [
        "The development of artificial intelligence has led to significant advances in various fields, "
        "including natural language processing, computer vision, and robotics. These technologies "
        "continue to evolve and shape our future in unprecedented ways.",
        
        "Machine learning algorithms have demonstrated remarkable capabilities in pattern recognition "
        "and data analysis. By processing vast amounts of information, these systems can identify "
        "complex relationships and make accurate predictions.",
        
        "Deep neural networks, inspired by the human brain's architecture, have revolutionized "
        "the field of artificial intelligence. Their ability to learn hierarchical representations "
        "has enabled breakthroughs in areas such as speech recognition and image classification.",
        
        "The intersection of artificial intelligence and healthcare has opened new possibilities "
        "for medical diagnosis and treatment. AI-powered systems can analyze medical images, "
        "predict patient outcomes, and assist in drug discovery.",
    ]

def test_distillation():
    """
    Test the distillation pipeline with a small sample of texts
    """
    # Get sample texts
    sample_texts = get_sample_texts()
    
    # Split into train and validation
    split_idx = int(len(sample_texts) * 0.75)
    train_texts = sample_texts[:split_idx]
    val_texts = sample_texts[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_texts, chunk_size=128)
    val_dataset = TextDataset(val_texts, chunk_size=128)
    
    # Initialize trainer with smaller models for testing
    trainer = DistillationTrainer(
        teacher_model_name="gpt2",  # Using smaller model for testing
        student_model_name="gpt2",
        temperature=2.0,
        alpha=0.5,
        learning_rate=1e-4,
        max_length=128,         
        batch_size=2
    )
    
    # Train for just one epoch
    trainer.train(train_dataset, val_dataset, num_epochs=1)
    
    return trainer

# Updated main script
if __name__ == "__main__":
    from trainer import DistillationTrainer 

    # # For debugging
    # trainer = test_distillation()
    # import sys
    # sys.exit(1)
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model_name="EleutherAI/gpt-neo-2.7B",
        student_model_name="gpt2",
        temperature=2.0,
        alpha=0.5,
        learning_rate=3e-4,
        max_length=128,
        batch_size=8
    )
    
    # Prepare data
    train_texts, val_texts = prepare_wikitext_data(split_ratio=0.9)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, chunk_size=trainer.max_length)
    val_dataset = TextDataset(val_texts, chunk_size=trainer.max_length)

    # Train the model
    trainer.train(train_dataset, val_dataset, num_epochs=3)

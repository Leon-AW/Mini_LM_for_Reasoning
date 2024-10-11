from transformers import Trainer, TrainingArguments
from models.reformer_model import get_reformer_model
from utils.data_utils import load_dataset

def train():
    model = get_reformer_model()
    train_dataset, val_dataset = load_dataset()

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    train()
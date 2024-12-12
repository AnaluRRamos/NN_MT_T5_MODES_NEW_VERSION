from src.preprocess import create_dataloaders

def load_data(data_dir, tokenizer, batch_size, num_workers=4):
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return train_dataloader, val_dataloader, test_dataloader

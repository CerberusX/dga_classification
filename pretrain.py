import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import argparse
from dataset import MLMDataset, preprocess_urls

def pretrain(args):
    # 读取Excel文件
    df = pd.read_excel(args.excel_file, header=None, names=['data'])
    df[['no', 'uri', 'type', 'label']] = df['data'].str.split(',', expand=True)
    urls = df['type'].tolist()
    labels = df['label'].tolist()
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    tokenized_urls = preprocess_urls(urls, tokenizer)

    mlm_dataset = MLMDataset(tokenized_urls, tokenizer)

    model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model_path)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=10_000,
        save_total_limit=2,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mlm_dataset
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_file', type=str, required=True, help='Path to the Excel file containing URL data')
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='Path to the pretrained BERT model')
    parser.add_argument('--output_dir', type=str, default='./mlm_trained_model', help='Directory to save the pretrained model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')

    args = parser.parse_args()
    pretrain(args)

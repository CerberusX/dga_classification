import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
from dataset import ClassificationDataset, preprocess_urls

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def finetune(args):
    # 读取微调数据集
    df = pd.read_excel(args.finetune_file, header=None, names=['data'])
    df[['isDGA', 'domain', 'host', 'subclass']] = df['data'].str.split(',', expand=True)

    finetune_urls = df['domain'].tolist()
    finetune_labels = df['subclass'].astype('category').cat.codes.tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    tokenized_finetune_urls = preprocess_urls(finetune_urls, tokenizer)

    classification_dataset = ClassificationDataset(tokenized_finetune_urls, finetune_labels, tokenizer)
    classification_dataloader = DataLoader(classification_dataset, batch_size=args.batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_path, num_labels=len(set(finetune_labels)))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=classification_dataset,
        eval_dataset=classification_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune_file', type=str, required=True, help='Path to the Excel file containing fine-tune data')
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='Path to the pretrained BERT model')
    parser.add_argument('--output_dir', type=str, default='./classification_results', help='Directory to save the fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')

    args = parser.parse_args()
    finetune(args)

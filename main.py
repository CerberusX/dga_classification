import os
import argparse

def main(args):
    # 自监督预训练
    os.system(f"python pretrain.py --excel_file {args.excel_file} --pretrained_model_path {args.pretrained_model_path} --output_dir {args.pretrain_output_dir} --epochs {args.pretrain_epochs} --batch_size {args.pretrain_batch_size}")

    # 微调
    os.system(f"python finetune.py --finetune_file {args.finetune_file} --pretrained_model_path {args.pretrain_output_dir} --output_dir {args.finetune_output_dir} --epochs {args.finetune_epochs} --batch_size {args.finetune_batch_size}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_file', type=str, required=True, help='Path to the Excel file containing URL data')
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='Path to the pretrained BERT model')
    parser.add_argument('--pretrain_output_dir', type=str, default='./mlm_trained_model', help='Directory to save the pretrained model')
    parser.add_argument('--pretrain_epochs', type=int, default=3, help='Number of pretraining epochs')
    parser.add_argument('--pretrain_batch_size', type=int, default=8, help='Batch size for pretraining')
    parser.add_argument('--finetune_file', type=str, required=True, help='Path to the Excel file containing fine-tune data')
    parser.add_argument('--finetune_output_dir', type=str, default='./classification_results', help='Directory to save the fine-tuned model')
    parser.add_argument('--finetune_epochs', type=int, default=3, help='Number of fine-tuning epochs')
    parser.add_argument('--finetune_batch_size', type=int, default=8, help='Batch size for fine-tuning')

    args = parser.parse_args()
    main(args)

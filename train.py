import argparse
import os
import pickle
import torch
import tqdm
import test_metrics
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from dataset_utils import SbicDataset
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_val_loss(val_dataloader, model):
    all_loss = 0
    model.eval()
    num_examples = 0
    predictions = []
    gold = []
    with torch.no_grad():
        for j, batch in enumerate(tqdm.tqdm(val_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            if j == 0:
                print(outputs)
            loss = outputs[0]
            all_loss += loss
            num_examples += len(input_ids)
            # print("===================================")
            # print(outputs[1].shape)
            # print(torch.argmax(outputs[1], dim=1).cpu().detach().tolist())
            predictions.extend(torch.argmax(outputs[1], dim=1).tolist())
            gold.extend(labels.cpu().tolist())
            torch.cuda.empty_cache()
    # print("Val====================")
    # print(gold[:50])
    # print(predictions[:50])
    f1 = f1_score(gold, predictions)
    return all_loss / num_examples, f1


def get_performance(test_dataloader, model):
    predictions = []
    gold = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            ids = batch["id"]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs[1], dim=1).tolist()
            # preds_with_ids = list(zip(preds, ids))
            predictions.extend(preds)
            gold.extend(labels.tolist())
    # print("GET PREDICTION =========================")
    # print(gold[50:])
    # print(predictions[50:])
    
    f1 = f1_score(gold, predictions)
    # print(f1)
    return f1

def train(output_dir, train_dataset_name, lr=5e-5, validate_every=10000, num_epochs=5, total_patience=20, batch_size=16, checkpoint_path=None):
    logging_file_path = os.path.join(output_dir, "logging")
    config_file_path = os.path.join(output_dir, "exp_config")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_file_path, "w") as config_writer:
        config_writer.write(str({"lr": lr, "validate_every": validate_every, "batch_size": batch_size}))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    train_dataset = SbicDataset("train", tokenizer, 32, train_dataset_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = SbicDataset("dev", tokenizer, 32, train_dataset_name)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = SbicDataset("test", tokenizer, batch_size=batch_size, dataset_name=train_dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    # Read from checkpoint
    if checkpoint_path is not None:
        model.from_pretrained(checkpoint_path)
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    best_perf = 0
    patience = 0

    i = 0 
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for j, batch in enumerate(tqdm.tqdm(train_loader)):
            i += j
            model.train()
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            if i % validate_every == 0 and i > 0:
                print(f"Validating at step {str(i)}")
                val_loss, val_perf = get_val_loss(val_dataloader, model)
                with open(logging_file_path, "a+") as logging_file_writer:
                    logging_file_writer.write(f"Training loss is {loss}")
                    logging_file_writer.write(f"Validation loss is {val_loss}")
                    logging_file_writer.write("\n")
                    logging_file_writer.write(f"Validation performance after {i} steps is {val_perf}")

                if best_val < val_loss:
                    best_val = val_loss
                else:
                    patience += 1
                if val_perf > best_perf:
                    best_perf = val_perf
                    model.save_pretrained(os.path.join(output_dir, "checkpoint_%s" % str(i)))
                if patience > total_patience:
                    break
    model.save_pretrained(os.path.join(output_dir, "final"))

    print("Evaluating on test set")
    f1 = get_performance(test_dataloader, model)
    print("F1 score: ", f1)
    # pickle.dump(test_predictions, open(os.path.join(output_dir, "test_predictions"), "wb"))
    # auc_results, fped_results, tped_resultss = test_metrics.get_predictions_and_evaluate(test_predictions)
    with open(os.path.join(output_dir,"test_metrics"), "w") as writer:
        writer.write(f"F1: {str(f1)}")
        # writer.write(f"FPED: {fped_results}")
        # writer.write(f"TPED: {tped_resultss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training stereotype classifier using social bias frames dataset')
    parser.add_argument('--train_dataset_name', type=str, default="all")
    parser.add_argument('--output_dir', type=str, default="/efs-storage/stereotype_classifier/")
    parser.add_argument('--lr', type=float, required=False, default=5e-6)
    parser.add_argument('--validate_every', type=int, required=False, default=1000)
    parser.add_argument('--num_epochs', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--checkpoint', type=str, required=False)
    args = parser.parse_args()
    train(args.output_dir,
          args.train_dataset_name,
          lr=args.lr,
          validate_every=args.validate_every,
          num_epochs=args.num_epochs,
          batch_size=args.batch_size,
          checkpoint_path=args.checkpoint)



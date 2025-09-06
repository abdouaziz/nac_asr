from nacasr import NACASR
from nacasr import get_dataloader
from nacasr import TextTransform , compute_input_lengths
import evaluate
import torch
import torch.nn.functional as F

from torch.nn import CTCLoss

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import time
import math



loss_fn = CTCLoss(blank=0, zero_infinity=True)
text_transform = TextTransform("abdouaziiz/new_benchmark_wolof")




def training_step(model, train_dataloader, optimizer, scheduler, device , loss_fn=loss_fn):

    model.train()

    for batch in train_dataloader:

        input_values = batch["input_values"].to(device)

        labels = batch["labels"].to(device)

        input_lengths = compute_input_lengths(input_values, in_dim=256)

        labels_lengths = (labels != -100).sum(-1)

        outputs = model(input_values)

        actual_output_lengths = torch.full((outputs.shape[0],), outputs.shape[1], dtype=torch.long)

        log_probs = F.log_softmax(outputs, dim=-1).transpose(0, 1)

        loss = loss_fn(log_probs, labels, actual_output_lengths, labels_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        return loss.item()



def eval_step(model, device, test_loader, criterion):
 
    model.eval()
    test_loss = 0
    test_wer = []

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            input_values = _data["input_values"].to(device)
            labels = _data["labels"].to(device)

            input_lengths = compute_input_lengths(input_values, in_dim=256)

            labels_lengths = (labels != -100).sum(-1)

            outputs = model(input_values)

            actual_output_lengths = torch.full((outputs.shape[0],), outputs.shape[1], dtype=torch.long)

            log_probs = F.log_softmax(outputs, dim=-1).transpose(0, 1)

            loss = criterion(log_probs, labels, actual_output_lengths, labels_lengths)

            test_loss += loss.item()

            decoded_preds, decoded_labels = greedyDecoder(
                log_probs.transpose(0, 1), labels, blank_id=0
            )

            wer_metric = evaluate.load("wer")
            wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
            test_wer.append(wer)



    return test_loss / len(test_loader), sum(test_wer) / len(test_wer)




def train_model(
    model,
    train_dataloader,
    test_dataloader,
    device,
    epochs=10,
    lr=1e-3,
    output_dir="./nacasr_model",
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    total_steps = len(train_dataloader) * epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="linear",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    best_wer = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        start_time = time.time()

        train_loss = training_step(model, train_dataloader, optimizer, scheduler, device)

        val_loss, val_wer = eval_step(model, device, test_dataloader, loss_fn)

        end_time = time.time()

        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WER: {val_wer:.4f} | Time: {int(epoch_mins)}m {int(epoch_secs)}s"
        )

        if val_wer < best_wer:
            best_wer = val_wer
            save_model(model, output_dir)

    print("Training complete.")



if __name__ == "__main__":
    dataloader = get_dataloader("abdouaziiz/new_benchmark_wolof", batch_size=2, num_workers=4)
    for batch in dataloader:
        print(batch)
        break
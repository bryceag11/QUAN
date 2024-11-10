# utils/torch_utils

from torch.cuda.amp import autocast, GradScaler

def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    for batch in dataloader:
        images = batch['images'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

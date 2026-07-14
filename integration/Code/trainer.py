"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
import copy

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device).view(-1).long()
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / total, (correct / total) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).view(-1).long()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return running_loss / total, (correct / total) * 100


    def fit(self, train_loader, val_loader, epochs, patience=7):
        print("\n Starting Training Routine...")
        print("-" * 50)
        best_state = None
        best_val = float("inf")
        current_patience = 0
        train_losses, val_losses = [], [] # store losses for plotting 
        best_epoch = -1 

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())    
                current_patience = 0
                best_epoch = epoch

            else: current_patience += 1
            if current_patience >= patience: # early stopping
                print("early stopping")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        print(f"best epoch: {best_epoch}, best val loss: {best_val:.6f}")
        print("-" * 50)
        print("Training Complete!")
        return train_losses, val_losses

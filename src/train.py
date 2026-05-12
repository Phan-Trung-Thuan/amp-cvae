import torch
import torch.optim as optim
from dataset import get_dataloaders
from models import AMPCVAE
from loss import MultiTaskCVAELoss
import os

def train_cvae(csv_path, epochs=10, batch_size=32, seq_type='lstm', lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} using {seq_type} architecture...")
    
    # 1. Get Data
    train_loader, val_loader, vocab, struct_enc, act_bin, config = get_dataloaders(csv_path, batch_size=batch_size)
    
    # 2. Initialize Model
    model = AMPCVAE(
        vocab_size=config['vocab_size'],
        seq_type=seq_type,
        struct_classes=config['struct_classes'],
        activity_classes=config['activity_classes']
    ).to(device)
    
    # 3. Initialize Loss and Optimizer
    criterion = MultiTaskCVAELoss(
        kl_weight=0.1, # Beta-VAE parameter tuning might be required
        pad_idx=config['pad_idx']
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_metrics = {}
        
        for batch in train_loader:
            seq = batch['sequence'].to(device)
            charge = batch['charge'].to(device)
            hydro = batch['hydrophobicity'].to(device)
            struct = batch['structure'].to(device)
            activity = batch['activities'].to(device)
            
            optimizer.zero_grad()
            
            # Encoder sees full sequence
            x_enc = seq
            
            # Decoder input should be shifted right (starts with SOS, misses last token)
            x_dec_in = seq[:, :-1]
            
            # Target is predicting the next token
            x_target = seq[:, 1:]
            
            recon_logits, mu, logvar, p_charge, p_hydro, p_struct, p_act = model(x_enc, x_dec_in)
            
            loss, losses_dict = criterion(
                recon_logits, x_target, mu, logvar,
                p_charge, charge,
                p_hydro, hydro,
                p_struct, struct,
                p_act, activity
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            for k, v in losses_dict.items():
                train_metrics[k] = train_metrics.get(k, 0.0) + v.item()
                
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_metrics = {}
        with torch.no_grad():
            for batch in val_loader:
                seq = batch['sequence'].to(device)
                charge = batch['charge'].to(device)
                hydro = batch['hydrophobicity'].to(device)
                struct = batch['structure'].to(device)
                activity = batch['activities'].to(device)
                
                x_enc = seq
                x_dec_in = seq[:, :-1]
                x_target = seq[:, 1:]
                
                recon_logits, mu, logvar, p_charge, p_hydro, p_struct, p_act = model(x_enc, x_dec_in)
                
                loss, losses_dict = criterion(
                    recon_logits, x_target, mu, logvar,
                    p_charge, charge,
                    p_hydro, hydro,
                    p_struct, struct,
                    p_act, activity
                )
                
                val_loss += loss.item()
                for k, v in losses_dict.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v.item()
                    
        # Average metrics
        train_loss /= len(train_loader)
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        
        for k in train_metrics: train_metrics[k] /= len(train_loader)
        if len(val_loader) > 0:
            for k in val_metrics: val_metrics[k] /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  [Train] Total Loss: {train_loss:.4f} | " + " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
        if len(val_loader) > 0:
            print(f"  [Val]   Total Loss: {val_loss:.4f} | " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
        
    print("Training complete!")
    
    # Save the model
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"amp_cvae_{seq_type}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model, vocab, struct_enc, act_bin

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "aps_database.csv")
    if os.path.exists(csv_file):
        train_cvae(csv_file, epochs=2, batch_size=8, seq_type='lstm')
    else:
        print("Data file missing.")

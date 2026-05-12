import torch
import torch.optim as optim
from dataset import get_dataloaders
from models import AMPCVAE
from loss import MultiTaskCVAELoss
import os

def train_cvae(csv_path, epochs=100, batch_size=32, seq_type='lstm', lr=1e-3, device_id=None):
    if device_id is not None and torch.cuda.device_count() > device_id:
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} using {seq_type} architecture for {epochs} epochs...")
    
    # 1. Get Data
    train_loader, val_loader, vocab, struct_enc, act_bin, config = get_dataloaders(csv_path, batch_size=batch_size)
    print(f"[{seq_type.upper()}] Dataset loaded: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")
    
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
        charge_weight=0.1,  # Scale down charge loss (range ~ -12 to 64)
        hydro_weight=0.01,  # Scale down hydro loss (range ~ 0 to 100)
        pad_idx=config['pad_idx']
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # History dictionary to store metrics
    history = {
        'train_total_loss': [], 'val_total_loss': [],
        'train_recon_loss': [], 'val_recon_loss': [],
        'train_kl_loss': [], 'val_kl_loss': [],
        'train_charge_loss': [], 'val_charge_loss': [],
        'train_hydro_loss': [], 'val_hydro_loss': [],
        'train_struct_loss': [], 'val_struct_loss': [],
        'train_activity_loss': [], 'val_activity_loss': []
    }
    
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
            
        # Store in history
        history['train_total_loss'].append(train_loss)
        history['val_total_loss'].append(val_loss)
        for k in ['recon_loss', 'kl_loss', 'charge_loss', 'hydro_loss', 'struct_loss', 'activity_loss']:
            history[f'train_{k}'].append(train_metrics[k])
            if len(val_loader) > 0:
                history[f'val_{k}'].append(val_metrics[k])

        # Print progress every 10 epochs or on the first/last epoch to avoid too much output
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            val_str = f" | Val Loss: {val_loss:.4f}" if len(val_loader) > 0 else ""
            print(f"[{seq_type.upper()}] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}{val_str}")
        
    print(f"Training complete for {seq_type}!")
    
    # Save the model
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"amp_cvae_{seq_type}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model, vocab, struct_enc, act_bin, history

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "aps_database.csv")
    if os.path.exists(csv_file):
        train_cvae(csv_file, epochs=2, batch_size=8, seq_type='lstm')
    else:
        print("Data file missing.")

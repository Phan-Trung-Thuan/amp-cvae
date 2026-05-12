import torch
import torch.nn as nn

class MultiTaskCVAELoss(nn.Module):
    """
    Combines Reconstruction Loss, KL-Divergence, MSE for continuous properties, 
    Categorical CE for structure, and BCE for activities.
    """
    def __init__(self, kl_weight=1.0, charge_weight=1.0, hydro_weight=1.0, struct_weight=1.0, activity_weight=1.0, pad_idx=0):
        super().__init__()
        self.kl_weight = kl_weight
        self.charge_weight = charge_weight
        self.hydro_weight = hydro_weight
        self.struct_weight = struct_weight
        self.activity_weight = activity_weight
        
        # Losses
        self.recon_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.mse_loss_fn = nn.MSELoss()
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, recon_logits, target_seq, mu, logvar, 
                pred_charge, target_charge,
                pred_hydro, target_hydro,
                pred_struct, target_struct,
                pred_activity, target_activity):
        
        # 1. Reconstruction Loss
        # recon_logits: (batch, seq_len, vocab_size)
        # target_seq: (batch, seq_len)
        recon_loss = self.recon_loss_fn(recon_logits.reshape(-1, recon_logits.size(-1)), target_seq.reshape(-1))
        
        # 2. KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0) # Average over batch
        
        # 3. Regression Losses
        charge_loss = self.mse_loss_fn(pred_charge, target_charge)
        hydro_loss = self.mse_loss_fn(pred_hydro, target_hydro)
        
        # 4. Classification Losses
        struct_loss = self.ce_loss_fn(pred_struct, target_struct)
        activity_loss = self.bce_loss_fn(pred_activity, target_activity)
        
        # Total Loss
        total_loss = recon_loss + \
                     self.kl_weight * kl_loss + \
                     self.charge_weight * charge_loss + \
                     self.hydro_weight * hydro_loss + \
                     self.struct_weight * struct_loss + \
                     self.activity_weight * activity_loss
                     
        losses = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'charge_loss': charge_loss,
            'hydro_loss': hydro_loss,
            'struct_loss': struct_loss,
            'activity_loss': activity_loss
        }
                     
        return total_loss, losses

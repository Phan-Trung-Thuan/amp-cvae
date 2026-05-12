import torch
import torch.nn.functional as F
from models import AMPCVAE
import os

def generate_peptides(model, vocab, target_charge, target_hydro, target_struct_idx, target_activity_vec, 
                      num_samples=5, max_len=100, temperature=1.0):
    """
    Standalone function to input target properties + random noise to generate new peptides.
    """
    device = next(model.parameters()).device
    model.eval()
    
    generated_sequences = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # 1. Sample Z from N(0, I)
            latent_dim = model.encoder.fc_mu.out_features
            z = torch.randn(1, latent_dim).to(device)
            
            # 2. Prepare Condition Vector
            charge = torch.tensor([[target_charge]], dtype=torch.float32).to(device)
            hydro = torch.tensor([[target_hydro]], dtype=torch.float32).to(device)
            
            # Structure probabilities (one-hot for the target)
            struct_probs = torch.zeros(1, model.struct_classes).to(device)
            struct_probs[0, target_struct_idx] = 1.0
            
            # Activity probabilities
            act_probs = torch.tensor([target_activity_vec], dtype=torch.float32).to(device)
            
            cond_vec = torch.cat([z, charge, hydro, struct_probs, act_probs], dim=-1)
            
            # 3. Autoregressive Generation
            # Start with <SOS>
            input_seq = torch.tensor([[vocab.stoi[vocab.sos_token]]], dtype=torch.long).to(device)
            
            generated = []
            
            for _ in range(max_len):
                # Pass current sequence through decoder
                logits = model.decoder(input_seq, cond_vec)
                
                # Get prediction for next token (last timestep)
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == vocab.stoi[vocab.eos_token]:
                    break
                    
                generated.append(next_token.item())
                input_seq = torch.cat([input_seq, next_token], dim=1)
                
            generated_sequences.append(vocab.decode(generated))
            
    return generated_sequences

if __name__ == "__main__":
    print("Testing Generation Script Structure...")
    print("Run train.py to train the model, load it here, then call generate_peptides().")
    # Example logic:
    # model.load_state_dict(torch.load('checkpoints/amp_cvae_lstm.pth'))
    # new_peptides = generate_peptides(model, vocab, target_charge=2.0, target_hydro=50.0, target_struct_idx=1, target_activity_vec=[1, 0, ...])
    # print(new_peptides)

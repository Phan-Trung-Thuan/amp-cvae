import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer architectures.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SequenceEncoder(nn.Module):
    """
    Base sequence encoder block that can easily swap between RNN, LSTM, GRU, and Transformer.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, seq_type='lstm', dropout=0.1):
        super().__init__()
        self.seq_type = seq_type.lower()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        if self.seq_type in ['rnn', 'lstm', 'gru']:
            rnn_class = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.seq_type]
            self.rnn = rnn_class(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.out_dim = hidden_dim
        elif self.seq_type == 'transformer':
            self.pos_encoder = PositionalEncoding(embed_dim, dropout)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=hidden_dim, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.out_dim = embed_dim
        else:
            raise ValueError(f"Unknown seq_type: {seq_type}. Choose from rnn, lstm, gru, transformer.")

    def forward(self, x):
        # x: (batch_size, seq_len)
        emb = self.embed(x) # (batch_size, seq_len, embed_dim)
        
        if self.seq_type in ['rnn', 'lstm', 'gru']:
            out, _ = self.rnn(emb)
            # Pool the output across sequence length to get a fixed-size representation.
            # Using mean pooling.
            seq_repr = out.mean(dim=1) 
        elif self.seq_type == 'transformer':
            emb = self.pos_encoder(emb)
            out = self.transformer(emb) 
            seq_repr = out.mean(dim=1)
            
        return seq_repr


class AMPEncoder(nn.Module):
    """
    The Encoder network for Purpose 1.
    Outputs a 768-dim latent space, plus predictions for 5 property heads.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, seq_type='lstm', 
                 latent_dim=768, struct_classes=7, activity_classes=27):
        super().__init__()
        self.seq_encoder = SequenceEncoder(vocab_size, embed_dim, hidden_dim, num_layers, seq_type)
        
        enc_out_dim = self.seq_encoder.out_dim
        
        # 1. Z mean and Z logvar for conditional VAE
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        
        # 2. Regressions
        self.fc_charge = nn.Linear(enc_out_dim, 1)
        self.fc_hydro = nn.Linear(enc_out_dim, 1)
        
        # 3. Classifications
        self.fc_struct = nn.Linear(enc_out_dim, struct_classes)
        self.fc_activity = nn.Linear(enc_out_dim, activity_classes)

    def forward(self, x):
        seq_repr = self.seq_encoder(x)
        
        mu = self.fc_mu(seq_repr)
        logvar = self.fc_logvar(seq_repr)
        
        charge = self.fc_charge(seq_repr)
        hydro = self.fc_hydro(seq_repr)
        struct = self.fc_struct(seq_repr)
        activity = self.fc_activity(seq_repr) # Outputs logits, BCEWithLogitsLoss will handle sigmoid later
        
        return mu, logvar, charge, hydro, struct, activity


class SequenceDecoder(nn.Module):
    """
    Base sequence decoder block that handles 804-dim conditional injection.
    """
    def __init__(self, vocab_size, cond_dim=804, embed_dim=128, hidden_dim=256, num_layers=2, seq_type='lstm', dropout=0.1):
        super().__init__()
        self.seq_type = seq_type.lower()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # We concatenate the condition vector to the embedding of each token
        dec_input_dim = embed_dim + cond_dim
        
        if self.seq_type in ['rnn', 'lstm', 'gru']:
            rnn_class = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.seq_type]
            self.rnn = rnn_class(dec_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc_out = nn.Linear(hidden_dim, vocab_size)
            
        elif self.seq_type == 'transformer':
            # Project concatenated input back to hidden dimension
            self.proj_in = nn.Linear(dec_input_dim, hidden_dim)
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
            decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*2, batch_first=True)
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.fc_out = nn.Linear(hidden_dim, vocab_size)
        else:
            raise ValueError(f"Unknown seq_type: {seq_type}")

    def forward(self, trg_seq, cond_vec):
        # trg_seq: (batch_size, seq_len)
        # cond_vec: (batch_size, cond_dim)
        
        batch_size, seq_len = trg_seq.size()
        emb = self.embed(trg_seq) # (batch_size, seq_len, embed_dim)
        
        # Expand cond_vec to match seq_len: (batch_size, seq_len, cond_dim)
        cond_repeated = cond_vec.unsqueeze(1).repeat(1, seq_len, 1) 
        
        # Inject conditions into every step of the decoder
        dec_in = torch.cat([emb, cond_repeated], dim=-1) 
        
        if self.seq_type in ['rnn', 'lstm', 'gru']:
            out, _ = self.rnn(dec_in) 
            logits = self.fc_out(out)
            
        elif self.seq_type == 'transformer':
            # Causal mask for autoregressive generation
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(trg_seq.device)
            
            dec_in = self.proj_in(dec_in)
            dec_in = self.pos_encoder(dec_in)
            
            # Since we injected the condition directly into the input sequence, we don't necessarily 
            # need an encoder memory output for the TransformerDecoder. We supply a dummy zero memory.
            memory_dummy = torch.zeros(batch_size, 1, dec_in.size(2)).to(trg_seq.device)
                
            out = self.transformer(tgt=dec_in, memory=memory_dummy, tgt_mask=mask)
            logits = self.fc_out(out)
            
        return logits


class AMPCVAE(nn.Module):
    """
    The Complete Conditional VAE Architecture assembling Encoder and Decoder.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, seq_type='lstm', 
                 latent_dim=768, struct_classes=7, activity_classes=27):
        super().__init__()
        
        self.encoder = AMPEncoder(vocab_size, embed_dim, hidden_dim, num_layers, seq_type, 
                                  latent_dim, struct_classes, activity_classes)
        
        # cond_dim = latent_dim (768) + charge (1) + hydro (1) + struct (7) + activity (27) = 804
        self.cond_dim = latent_dim + 1 + 1 + struct_classes + activity_classes
        
        self.decoder = SequenceDecoder(vocab_size, self.cond_dim, embed_dim, hidden_dim, num_layers, seq_type)
        
        self.struct_classes = struct_classes
        self.activity_classes = activity_classes

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, enc_input, dec_input):
        """
        enc_input: full sequence
        dec_input: sequence shifted right (missing EOS or last token)
        """
        mu, logvar, charge, hydro, struct_logits, activity_logits = self.encoder(enc_input)
        
        z = self.reparameterize(mu, logvar)
        
        # To make decoder robust, we pass the softly predicted properties 
        # as conditional injections (differentiable).
        struct_probs = torch.softmax(struct_logits, dim=-1)
        activity_probs = torch.sigmoid(activity_logits)
        
        cond_vec = torch.cat([z, charge, hydro, struct_probs, activity_probs], dim=-1)
        
        # Decode conditioned on injected vector
        recon_logits = self.decoder(dec_input, cond_vec)
        
        return recon_logits, mu, logvar, charge, hydro, struct_logits, activity_logits

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

class AMPVocab:
    """
    Handles character-level tokenization for Amino Acid sequences.
    Standard 20 amino acids + special tokens.
    """
    def __init__(self):
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        
        # Standard 20 amino acids
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        
        self.itos = {
            0: self.pad_token,
            1: self.sos_token,
            2: self.eos_token,
            3: self.unk_token,
        }
        
        for i, aa in enumerate(self.amino_acids):
            self.itos[i + 4] = aa
            
        self.stoi = {v: k for k, v in self.itos.items()}
        self.vocab_size = len(self.itos)

    def encode(self, seq, add_special_tokens=True):
        tokens = [self.stoi.get(aa.upper(), self.stoi[self.unk_token]) for aa in seq]
        if add_special_tokens:
            tokens = [self.stoi[self.sos_token]] + tokens + [self.stoi[self.eos_token]]
        return tokens

    def decode(self, tokens, ignore_special=True):
        special_ids = {0, 1, 2, 3}
        decoded = []
        for t in tokens:
            if ignore_special and t in special_ids:
                continue
            decoded.append(self.itos.get(t, self.unk_token))
        return "".join(decoded)


class AMPDataset(Dataset):
    """
    Custom PyTorch Dataset for Antimicrobial Peptides.
    Provides tokens, regression targets, multiclass structure, and multilabel activities.
    """
    def __init__(self, df, vocab, struct_encoder, activity_binarizer, max_len=100):
        self.vocab = vocab
        self.max_len = max_len
        self.df = df.reset_index(drop=True)
        
        self.struct_encoder = struct_encoder
        self.activity_binarizer = activity_binarizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Sequence Tokenization (Tokens)
        seq = row['Sequence']
        encoded_seq = self.vocab.encode(seq)
        
        # Truncate or Pad to fixed max_len
        if len(encoded_seq) > self.max_len:
            encoded_seq = encoded_seq[:self.max_len]
        else:
            encoded_seq = encoded_seq + [self.vocab.stoi[self.vocab.pad_token]] * (self.max_len - len(encoded_seq))
            
        seq_tensor = torch.tensor(encoded_seq, dtype=torch.long)
        
        # 2. Net Charge (Integer Regression)
        charge = torch.tensor([float(row['Net charge'])], dtype=torch.float32)
        
        # 3. Hydrophobic residue % (Integer Regression)
        hydro = torch.tensor([float(row['Hydrophobic_perc'])], dtype=torch.float32)
        
        # 4. 3D Structure (Multiclass)
        # Using .transform() which returns array, get first element
        struct_idx = self.struct_encoder.transform([row['3D Structure']])[0]
        struct_tensor = torch.tensor(struct_idx, dtype=torch.long)
        
        # 5. Activities (Multilabel)
        activities = row['Activity_List']
        activity_vec = self.activity_binarizer.transform([activities])[0]
        activity_tensor = torch.tensor(activity_vec, dtype=torch.float32)
        
        return {
            'sequence': seq_tensor,
            'charge': charge,
            'hydrophobicity': hydro,
            'structure': struct_tensor,
            'activities': activity_tensor
        }


def preprocess_dataframe(csv_path):
    """
    Reads and cleans the APS database.
    """
    df = pd.read_csv(csv_path)
    
    # Clean Sequence (drop NaN sequences)
    df = df.dropna(subset=['Sequence']).copy()
    df['Sequence'] = df['Sequence'].astype(str).str.strip()
    
    # 1. Clean Hydrophobic residue %
    def clean_hydro(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, str):
            return float(x.replace('%', '').strip())
        return float(x)
        
    df['Hydrophobic_perc'] = df['Hydrophobic residue%'].apply(clean_hydro)
    
    # 2. Clean Net charge
    df['Net charge'] = pd.to_numeric(df['Net charge'], errors='coerce').fillna(0.0)
    
    # 3. Clean 3D Structure
    # Mapping slight string variations to unified categories to strictly enforce 7 classes if possible, 
    # but we'll use a lower() and strip() as baseline cleaning.
    df['3D Structure'] = df['3D Structure'].astype(str).str.lower().str.strip()
    
    # 4. Clean Activities
    def parse_activities(act_str):
        if pd.isna(act_str):
            return []
        # Split by comma, trim whitespace
        return [a.strip() for a in str(act_str).split(',') if a.strip()]
        
    df['Activity_List'] = df['Activity'].apply(parse_activities)
    
    return df

def get_dataloaders(csv_path, batch_size=64, max_len=100, val_split=0.1, random_seed=42):
    """
    Main entry point for Step 1. Preprocesses the data, builds vocab/encoders,
    and returns PyTorch DataLoaders.
    """
    df = preprocess_dataframe(csv_path)
    
    # Fit Struct Encoder
    struct_encoder = LabelEncoder()
    df['3D Structure'] = struct_encoder.fit_transform(df['3D Structure'])
    # Inverse transform so Dataset can transform them (or we could just use the encoded ints)
    # Reverting for consistency in dataset definition above
    df['3D Structure'] = struct_encoder.inverse_transform(df['3D Structure']) 
    
    # Fit Activity Binarizer
    act_binarizer = MultiLabelBinarizer()
    act_binarizer.fit(df['Activity_List'])
    
    # Vocab
    vocab = AMPVocab()
    
    # Train/Val Split
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=random_seed)
    
    train_dataset = AMPDataset(train_df, vocab, struct_encoder, act_binarizer, max_len=max_len)
    val_dataset = AMPDataset(val_df, vocab, struct_encoder, act_binarizer, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    config = {
        'vocab_size': vocab.vocab_size,
        'pad_idx': vocab.stoi[vocab.pad_token],
        'struct_classes': len(struct_encoder.classes_),
        'activity_classes': len(act_binarizer.classes_),
        'max_len': max_len
    }
    
    return train_loader, val_loader, vocab, struct_encoder, act_binarizer, config

if __name__ == "__main__":
    print("Testing Step 1: Data Preprocessing & Custom PyTorch Dataset/DataLoader")
    csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "aps_database.csv")
    
    if os.path.exists(csv_file):
        train_loader, val_loader, vocab, struct_enc, act_bin, config = get_dataloaders(csv_file, batch_size=4)
        
        print(f"Vocab Size: {config['vocab_size']}")
        print(f"3D Structure Classes ({config['struct_classes']}):", struct_enc.classes_)
        print(f"Activity Classes ({config['activity_classes']}):", act_bin.classes_[:5], "...")
        
        # Test 1 batch
        for batch in train_loader:
            print("\n--- Sample Batch Shapes ---")
            print("Sequence:", batch['sequence'].shape)
            print("Charge:", batch['charge'].shape)
            print("Hydrophobicity:", batch['hydrophobicity'].shape)
            print("Structure:", batch['structure'].shape)
            print("Activities:", batch['activities'].shape)
            
            print("\n--- Example Decoded Sequence ---")
            print(vocab.decode(batch['sequence'][0].tolist(), ignore_special=False))
            break
    else:
        print(f"CSV file not found at {csv_file}. Please ensure it's placed correctly.")

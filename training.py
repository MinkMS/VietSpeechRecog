import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
import os

# This program will download the dataset straight from hugging face. Just run the code.

# In case of not enough memory in C:\, use this to store the cache in D:\
# Run the code editor in administrator, and then create a folder named huggingface_cache in D:\
# os.environ["HF_HOME"] = "D:/huggingface_cache"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Handling GPU memo prob

# 1. Load the Dataset from Hugging Face
print("Loading dataset...")

# Load the dataset (maybe split the dataset)
dataset = load_dataset("doof-ferb/vlsp2020_vinai_100h", split="train") # full dataset

# dataset = load_dataset("doof-ferb/vlsp2020_vinai_100h", split="train[:x%]") # x% of the dataset, replace it

# Download, decode, and resample audio to 16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Detect the transcription key (It's not transcription)
transcript_key = [key for key in dataset.features.keys() if key != "audio"][0]
print(f"Detected transcription key: {transcript_key}")

# 2. Preprocess the Audio (not necessary, but recommended)
def preprocess_audio(example):
    
    # Convert the raw audio array to a float32 tensor
    speech_array = torch.tensor(example["audio"]["array"], dtype=torch.float32)
    sampling_rate = example["audio"]["sampling_rate"]

    # Resampling is likely unnecessary because of cast_column, but kept for safety
    if sampling_rate != 16000:
        resampler = T.Resample(sampling_rate, 16000)
        speech_array = resampler(speech_array)

    # Compute the Mel Spectrogram (using 128 mel bins)
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=128)
    
    # mel_transform returns a tensor of shape (channel, n_mels, time)
    mel_spec_tensor = mel_transform(speech_array).squeeze(0).transpose(0, 1)  # Now shape: (time, n_mels)
    mel_spec_tensor = mel_spec_tensor.to(torch.float32)  # Ensure dtype is float32
    example["mel_spec"] = mel_spec_tensor
    return example

print("Preprocessing audio...")
dataset = dataset.map(preprocess_audio)

# 3. Build the Vocabulary from Transcriptions
def build_vocab(dataset, transcript_key):
    vocab_set = set()
    
    for sample in dataset:
        # Update the set with all characters in the transcript
        vocab_set.update(list(sample[transcript_key]))
        
    # Reserve index 0 for the CTC blank token
    vocab = ["<blank>"] + sorted(list(vocab_set))
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    return vocab, char2idx, idx2char

print("Building vocabulary...")
vocab, char2idx, idx2char = build_vocab(dataset, transcript_key)
num_classes = len(vocab)
print("Number of classes (including blank):", num_classes)

# 4. Create a Collate Function for the DataLoader
def collate_fn(batch):
    mel_specs_list = []
    transcripts = []
    for sample in batch:
        mel = sample["mel_spec"]
        
        # Ensure mel_spec is a tensor
        if not isinstance(mel, torch.Tensor):
            mel = torch.tensor(mel, dtype=torch.float32)
        mel_specs_list.append(mel)
        transcripts.append(sample[transcript_key])

    # Record original time lengths for each mel spectrogram
    mel_lengths = [spec.size(0) for spec in mel_specs_list]
    
    # Pad mel spectrograms (resulting shape: (batch, max_time, n_mels))
    mel_specs_padded = nn.utils.rnn.pad_sequence(mel_specs_list, batch_first=True)

    # Encode transcripts into a flat list of indices
    targets = []
    target_lengths = []
    for t in transcripts:
        t_idx = [char2idx[c] for c in t]
        targets.append(torch.tensor(t_idx, dtype=torch.long))
        target_lengths.append(len(t_idx))
    targets = torch.cat(targets)

    return mel_specs_padded, torch.tensor(mel_lengths, dtype=torch.long), targets, torch.tensor(target_lengths, dtype=torch.long)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# 5. Define the CRNN Model (CNN + LSTM + FC with CTC Loss)
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            
            # Input shape: (batch, 1, time, 128)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (batch, 32, time/2, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # -> (batch, 64, time/4, 32)
        )
        # After pooling, frequency dimension becomes 128/2/2 = 32, channels = 64.
        # LSTM input size = 64 * 32 = 2048.
        self.rnn = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2,
                           bidirectional=True, batch_first=True)
        
        # Final fully-connected layer: output dimension = number of classes
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # -> (batch, 1, time, n_mels)
        x = self.cnn(x)     # -> (batch, 64, time/4, 32)
        batch_size, channels, time_steps, freq_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, time_steps, channels * freq_dim)
        x, _ = self.rnn(x)  # -> (batch, time/4, 256)
        x = self.fc(x)      # -> (batch, time/4, num_classes)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = CRNN(num_classes).to(device)

# 6. Set Up Training Components
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 7. Training Loop

num_epochs = 10 # Can reduce it for faster but less accurate training, recommended for tessting the code

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for mel_specs, mel_lengths, targets, target_lengths in train_loader:
        mel_specs = mel_specs.to(device)
        targets = targets.to(device)
        logits = model(mel_specs)
        logits = logits.permute(1, 0, 2)
        log_probs = nn.functional.log_softmax(logits, dim=2)
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, targets, mel_lengths // 4, target_lengths)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}") # Loss ban dau thuong la treen 2.5 va duoi 3.0.

print("Training complete!")

#Save the model
torch.save(model.state_dict(), "viet_asr_model.pth")
print("Model saved successfully!")

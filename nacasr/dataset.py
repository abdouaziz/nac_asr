import torch
import torch.nn as nn

from vector_quantize_pytorch import ResidualVQ

from torch.utils.data import Dataset, DataLoader
import librosa

from datasets import load_dataset, Audio




class DatasetError(Exception):
    pass


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, channels: int = 512):
        super().__init__()
        self.strides = [5, 2, 2, 2, 2, 2, 2]
        self.kernels = [10, 3, 3, 3, 3, 2, 2]
        self.channels = channels

        self.layers = nn.ModuleList()
        current_channels = in_channels

        for kernel_size, stride in zip(self.kernels, self.strides):
            self.layers.append(
                nn.Conv1d(
                    current_channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False,
                )
            )
            current_channels = channels

        self.layer_norm = nn.LayerNorm(channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Normalize input to zero mean and unit variance (wav2vec 2.0 paper)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)

        for layer in self.layers:
            x = layer(x)   
            x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)  
            x = self.gelu(x)  

        return x


class FeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=1, channels=256)
        self.quantizer = ResidualVQ(
            dim=256,
            codebook_size=256,
            num_quantizers=4,
            kmeans_init=True,
            kmeans_iters=10,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.transpose(1, 2)
        quantized, indices, commit_loss = self.quantizer(encoded)
        return quantized, indices, commit_loss


class TextTransform:
    def __init__(self, path_or_name):
        ds = load_dataset(path_or_name)
        ds = ds.cast_column("audio", Audio())
        self.vocab = " "
        for item in ds["train"]["transcription"]:
            self.vocab += item.lower() + " "
        self.vocab = set(self.vocab)
        self.vocab = sorted(self.vocab)
        self.vocab_dict = {v: k for k, v in enumerate(self.vocab)}

        self.char_map = {}
        self.index_map = {}
        for line in self.vocab_dict:
            ch, index = line, self.vocab_dict[line]
            self.char_map[ch] = index
            self.index_map[index] = ch

    def text_to_int(self, text):
        int_sequence = []
        for c in text:
            ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return "".join(string)
    
    def vocab_size(self):
        return len(self.vocab)


class NACDataset(Dataset):
    def __init__(self, path_or_name):
        self.ds = load_dataset(path_or_name)
        self.ds = self.ds.cast_column("audio", Audio())
        self.text_transform = TextTransform(path_or_name)
        self.feature_extractor = FeaturesExtractor()
        self.feature_extractor.eval()

        self.ds = self.ds.map(self.audioPreprocessor, num_proc=4)

        self.samples = self.ds["train"]

    @staticmethod
    def audioPreprocessor(batch):

        speech_array = batch["audio"]["array"]
        sampling_rate = batch["audio"]["sampling_rate"]

        if sampling_rate != 16000:
            resampled_speech = librosa.resample(
                speech_array, orig_sr=sampling_rate, target_sr=16000
            )

            batch["audio"]["array"] = resampled_speech

        return batch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get discrete audio features
        audio = self.samples[idx]["audio"]["array"]
        transcription = self.samples[idx]["transcription"].lower()

        transcription = self.text_transform.text_to_int(transcription)

        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        with torch.no_grad():
            quantized, _, _ = self.feature_extractor(audio_tensor.unsqueeze(0))

        return {
            "input_values": quantized.squeeze(0),
            "labels": torch.tensor(transcription, dtype=torch.long),
        }


def _collate_fn(batch):
    data = [item["input_values"] for item in batch]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    labels = [item["labels"] for item in batch]
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    return {"input_values": data, "labels": labels}


def get_dataloader(path_or_name, batch_size=8, shuffle=True, num_workers=0):
    try:

        dataset = NACDataset(path_or_name)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_fn,
            num_workers=num_workers, 
        )
    except Exception as e:
        raise DatasetError(f"Error from creation of dataloader: {e}")


# if __name__ == "__main__":
#     dataloader = get_dataloader("abdouaziiz/new_benchmark_wolof", batch_size=2, num_workers=4)
#     for batch in dataloader:
#         print(batch)
#         break

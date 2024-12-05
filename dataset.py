import torchvision
import torchvision.transforms as T 
from torchvision.utils import save_image
from librosa.filters import mel as librosa_mel_fn
import torch, torchaudio
import os
import pandas as pd
from huggingface_hub import snapshot_download, hf_hub_download
#from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn, Tensor, tensor, is_tensor, stack


IMAGE_AFTER_TEXT = False 

# character based tokenizer

def char_tokenize(
    text,
    device = None,
    offset = 0
) -> Tensor:
    return tensor([*map(ord, text)], device = device) + offset


def decode_chars(
    t,
    offset = 0,
) -> str:
    byte_list = (t - offset).clamp(min = 0, max = 127).tolist()
    return ''.join([*map(chr, byte_list)])

class MnistDataset(Dataset):
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST(
            './data/mnist',
            download = True
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = T.PILToTensor()(pil)
        output =  tensor(labels), (digit_tensor / 255).float()

        if not IMAGE_AFTER_TEXT:
            return output

        first, second = output
        return second, first

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

def collate_fn(data):
    data = [*map(list, data)]
    return data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mel_basis_cache = {}
hann_window_cache = {}

def get_bigvgan_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=None,
    center=False,
):  # Copy from https://github.com/NVIDIA/BigVGAN/tree/main
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=target_sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)  # TODO: why they need .float()?
        hann_window_cache[key] = torch.hann_window(win_length).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec

def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel

class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    ):
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"], print("We only support two extract mel backend: vocos or bigvgan")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram
        elif mel_spec_type == "bigvgan":
            self.extractor = get_bigvgan_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel

def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


extract_dir = "./LJSpeech-1.1"
# List the files in the extracted directory
dataset_dir = '/home/askhat.sametov/LJSpeech-1.1/LJSpeech-1.1'
extracted_files = os.listdir('/home/askhat.sametov/LJSpeech-1.1/LJSpeech-1.1')


lj = pd.read_csv(os.path.join(dataset_dir, 'metadata.csv'), sep='|', header=None, names=['ID', 'Transcription', 'Normalized_Transcription'])
lj = lj.dropna()
lj.info()

class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset,
        #durations = None,
        target_sample_rate = 24_000,
        hop_length = 256,
        n_mel_channels = 100,
        preprocessed_mel = False,
    ):
        self.data = custom_dataset
        #self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.preprocessed_mel = preprocessed_mel
        if not preprocessed_mel:
            self.mel_spectrogram = MelSpec(target_sample_rate=target_sample_rate, hop_length=hop_length, n_mel_channels=n_mel_channels)

    #def get_frame_len(self, index):
    #    if self.durations is not None:  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
    #        return self.durations[index] * self.target_sample_rate / self.hop_length
    #    return self.data[index]["duration"] * self.target_sample_rate / self.hop_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        audio_path = row["audio_path"]
        text = char_tokenize(row["text"])
        # text = row['text']
        #duration = row["duration"]

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])

        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            #if duration > 30 or duration < 0.3:
            #    return self.__getitem__((index + 1) % len(self.data))
            
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)
            
            mel_spec = self.mel_spectrogram(audio)
            #mel_spec = rearrange(mel_spec, '1 d t -> d t')
            mel_spec = mel_spec.squeeze(0)
            mel_spec = mel_spec.permute(1, 0)
        
        return text, mel_spec 

        # return dict(
        #     mel_spec = mel_spec,
        #     text = text,
        # )

# Function to convert the pandas DataFrame to the format required for the CustomDataset
audio_folder = os.path.join(dataset_dir, 'wavs')

def convert_pandas_to_custom_format(df, audio_folder):
    dataset = []
    
    for _, row in df.iterrows():
        #audio_path = os.path.join(audio_folder, df['ID'][0] + '.wav')
        audio_path = os.path.join(audio_folder, f"{row['ID']}.wav")
        
        dataset.append({
            'audio_path': audio_path,
            'text': row['Normalized_Transcription'],
            # 'duration': 2.5  # Optional
        })
    
    return dataset

def collate_fn(data):
    return [*map(list, data)]

def create_dataloader(dataset: Dataset, **kwargs) -> DataLoader:
    return DataLoader(dataset, collate_fn = collate_fn, **kwargs)

custom_data = convert_pandas_to_custom_format(lj, audio_folder)

train_data, test_data = train_test_split(custom_data, test_size=0.1, random_state=42)
train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)

# custom_mel_dataset = CustomDataset(custom_data) 
train_dataloader = create_dataloader(train_dataset, batch_size = 4, shuffle = True)
test_dataloader = create_dataloader(test_dataset, batch_size = 4, shuffle = False) 


import torch
import random
import json
import os
import numpy as np

node_dir = os.path.dirname(os.path.abspath(__file__))

def decode_audio(
    latents: torch.Tensor,
    vae_model: torch.nn.Module,
    chunked: bool = False,
    overlap: int = 32,
    chunk_size: int = 128
):
    downsampling_ratio = 2048
    io_channels = 2
    if not chunked:
        try:
            output = vae_model.decode_export(latents)
            return output
        except Exception as e:
            raise
    else:
        # Chunked decoding logic
        hop_size = chunk_size - overlap
        total_size = latents.shape[2]
        batch_size = latents.shape[0]
        chunks = []
        i = 0
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunk = latents[:, :, i : i + chunk_size]
            chunks.append(chunk)
        if i + chunk_size != total_size:
            # Final chunk
            chunk = latents[:, :, -chunk_size:]
            chunks.append(chunk)
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        # samples_per_latent is just the downsampling ratio
        samples_per_latent = downsampling_ratio
        # Create an empty waveform, we will populate it with chunks as decode them
        y_size = total_size * samples_per_latent
        y_final = torch.zeros((batch_size, io_channels, y_size)).to(latents.device)
        for i in range(num_chunks):
            x_chunk = chunks[i, :]
            try:
                y_chunk = vae_model.decode_export(x_chunk)
            except Exception as e:
                raise
            # figure out where to put the audio along the time domain
            if i == num_chunks - 1:
                # final chunk always goes at the end
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
            #  remove the edges of the overlaps
            ol = (overlap // 2) * samples_per_latent
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if i > 0:
                # no overlap for the start of the first chunk
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                # no overlap for the end of the last chunk
                t_end -= ol
                chunk_end -= ol
            # paste the chunked audio into our y_final output audio
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
        return y_final

def get_reference_latent(device: torch.device, max_frames: int):
    return torch.zeros(1, max_frames, 64).to(device)

def get_negative_style_prompt(device: torch.device):
    file_path = f"{node_dir}/vocal.npy"
    try:
        vocal_style = np.load(file_path)
    except Exception as e:
        raise

    vocal_style = torch.from_numpy(vocal_style).to(device)  # [1, 512]
    return vocal_style.half()

def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.strip()
    for line in lyrics.split("\n"):
        try:
            time, lyric = line[1:9], line[10:]
            mins, secs = time.split(":")
            secs = int(mins) * 60 + float(secs)
            lyrics_with_time.append((secs, lyric.strip()))
        except ValueError:
            continue
    return lyrics_with_time

class CNENTokenizer:
    def __init__(self):
        vocab_path = f"{node_dir}/g2p/g2p/vocab.json"
        try:
            with open(vocab_path, "r", encoding="utf-8") as file:
                self.phone2id: dict = json.load(file)["vocab"]
        except Exception as e:
            raise
            
        self.id2phone = {v: k for k, v in self.phone2id.items()}
        
        try:
            from g2p.g2p_generation import chn_eng_g2p
            self.tokenizer = chn_eng_g2p
        except Exception as e:
            raise

    def encode(self, text: str):
        try:
            phone, token = self.tokenizer(text)
            return [x + 1 for x in token]
        except Exception as e:
            print(f"Text encoding failed: {str(e)}")
            raise

    def decode(self, token: list):
        try:
            return "|".join([self.id2phone[x - 1] for x in token])
        except Exception as e:
            raise

def get_lrc_token(
    max_frames: int,
    text: str,
    tokenizer: CNENTokenizer,
    device: torch.device
):
    # Audio processing parameters
    lyrics_shift = 0
    sampling_rate = 44100
    downsample_rate = 2048
    max_secs = max_frames / (sampling_rate / downsample_rate)

    # Token configuration
    comma_token_id = 1
    period_token_id = 2

    lrc_with_time = parse_lyrics(text)

    modified_lrc_with_time = []
    for i in range(len(lrc_with_time)):
        time, line = lrc_with_time[i]
        try:
            line_token = tokenizer.encode(line)
            modified_lrc_with_time.append((time, line_token))
        except Exception as e:
            raise
    lrc_with_time = modified_lrc_with_time

    lrc_with_time = [
        (time_start, line)
        for (time_start, line) in lrc_with_time
        if time_start < max_secs
    ]
    # lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time

    normalized_start_time = 0.0

    lrc = torch.zeros((max_frames,), dtype=torch.long)

    tokens_count = 0
    last_end_pos = 0
    for time_start, line in lrc_with_time:
        tokens = [
            token if token != period_token_id else comma_token_id for token in line
        ] + [period_token_id]
        tokens = torch.tensor(tokens, dtype=torch.long)
        num_tokens = tokens.shape[0]

        gt_frame_start = int(time_start * sampling_rate / downsample_rate)

        frame_shift = random.randint(int(lyrics_shift), int(lyrics_shift))

        frame_start = max(gt_frame_start - frame_shift, last_end_pos)
        frame_len = min(num_tokens, max_frames - frame_start)

        lrc[frame_start : frame_start + frame_len] = tokens[:frame_len]

        tokens_count += num_tokens
        last_end_pos = frame_start + frame_len

    lrc_emb = lrc.unsqueeze(0).to(device)

    normalized_start_time = torch.tensor(normalized_start_time).unsqueeze(0).to(device)
    if device == "cuda":
        normalized_start_time = normalized_start_time.half()
    else:
        normalized_start_time = normalized_start_time.float()

    return lrc_emb, normalized_start_time

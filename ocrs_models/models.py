from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                # Equivalent to "same" padding for a kernel size of 3.
                # PyTorch's ONNX export doesn't support the "same" keyword.
                padding=(1, 1),
                bias=False,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.seq(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.seq = nn.Sequential(
            DepthwiseConv(in_channels, out_channels),
            DepthwiseConv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor):
        return self.seq(x)


class Down(nn.Module):
    """
    Downscaling module in U-Net model.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.seq = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor):
        return self.seq(x)


class Up(nn.Module):
    """
    Upscaling module in U-Net model.

    This upscales a feature map from the previous "up" stage of the network
    and combines it with a feature map coming across from a "down" stage.
    """

    def __init__(self, in_up_channels: int, in_cross_channels: int, out_channels: int):
        """
        :param in_up_channels: Channels in inputs to be upscaled
        :param in_cross_channels: Channels in inputs to be concatenated with upscaled input
        """
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_up_channels, out_channels, kernel_size=3, stride=2
        )
        self.contract = DoubleConv(out_channels + in_cross_channels, out_channels)

    def forward(self, x_to_upscale: torch.Tensor, x: torch.Tensor):
        upscaled = self.up(x_to_upscale)

        # `x_to_upscale` is assumed to be half the resolution of `x`. When
        # it is conv-transposed the result can be 1 pixel taller/wider than `x`.
        # Trim the right/bottom edges to make the images the same size.
        upscaled = upscaled[:, :, 0 : x.shape[2], 0 : x.shape[3]]

        combined = torch.cat((upscaled, x), dim=1)
        return self.contract(combined)


class DetectionModel(nn.Module):
    """
    Text detection model.

    This uses a U-Net-like architecture. See https://arxiv.org/abs/1505.04597.

    It expects a greyscale image as input and outputs a text/not-text
    segmentation mask.
    """

    def __init__(self):
        super().__init__()

        # Number of feature channels at each size level in the network.
        #
        # The U-Net paper uses 64 for the first level. This model uses a
        # reduced scale to cut down the parameter count.
        #
        # depth_scale = [64, 128, 256, 512, 1024]
        depth_scale = [8, 16, 32, 32, 64, 128, 256]
        self.depth_scale = depth_scale

        self.in_conv = DoubleConv(1, depth_scale[0])

        self.down = nn.ModuleList()
        for i in range(len(depth_scale) - 1):
            self.down.append(Down(depth_scale[i], depth_scale[i + 1]))

        self.up = nn.ModuleList()
        for i in range(len(depth_scale) - 1):
            self.up.append(Up(depth_scale[i + 1], depth_scale[i], depth_scale[i]))

        n_masks = 1  # Output masks to generate
        self.out_conv = nn.Sequential(
            nn.Conv2d(depth_scale[0], n_masks, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)

        x_down: list[torch.Tensor] = []
        for i, down_op in enumerate(self.down):
            prev_down = x if i == 0 else x_down[-1]
            x_down.append(down_op(prev_down))

        x_up = x_down[-1]
        for i, up_op in reversed(list(enumerate(self.up))):
            x_up = up_op(x_up, x if i == 0 else x_down[i - 1])

        return self.out_conv(x_up)


class DetectionModelV2(nn.Module):
    """
    Text detection model with Feature Pyramid Network (FPN).

    Extends the U-Net encoder with FPN-style lateral connections and
    multi-scale predictions for improved detection of text at varying sizes.

    Outputs ``n_masks`` channels:
    - Channel 0: text/not-text segmentation mask (same as DetectionModel)
    - Channel 1: column/paragraph separator mask
    """

    def __init__(self, n_masks: int = 2):
        super().__init__()

        depth_scale = [8, 16, 32, 32, 64, 128, 256]
        self.depth_scale = depth_scale
        fpn_channels = 64

        # Encoder (same as DetectionModel)
        self.in_conv = DoubleConv(1, depth_scale[0])
        self.down = nn.ModuleList()
        for i in range(len(depth_scale) - 1):
            self.down.append(Down(depth_scale[i], depth_scale[i + 1]))

        # FPN lateral connections: project each encoder level to fpn_channels
        self.fpn_lateral = nn.ModuleList(
            [nn.Conv2d(d, fpn_channels, kernel_size=1) for d in depth_scale]
        )

        # FPN smoothing convolutions (3x3 after element-wise addition)
        self.fpn_smooth = nn.ModuleList(
            [
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
                for _ in depth_scale
            ]
        )

        # Multi-scale prediction heads at 1/4, 1/8, 1/16 resolution
        # (indices 2, 3, 4 in encoder feature list)
        self.scale_heads = nn.ModuleList(
            [nn.Conv2d(fpn_channels, n_masks, kernel_size=1) for _ in range(3)]
        )

        # Full-resolution prediction head from the finest FPN level
        self.out_conv = nn.Conv2d(fpn_channels, n_masks, kernel_size=1)

        # Learnable weights for fusing multi-scale predictions
        self.scale_weights = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder forward pass: collect features at each level
        encoder_features = [self.in_conv(x)]
        for down_op in self.down:
            encoder_features.append(down_op(encoder_features[-1]))

        # FPN top-down pathway
        fpn_features = [None] * len(encoder_features)
        fpn_features[-1] = self.fpn_lateral[-1](encoder_features[-1])

        for i in range(len(encoder_features) - 2, -1, -1):
            upsampled = F.interpolate(
                fpn_features[i + 1],
                size=encoder_features[i].shape[2:],
                mode="nearest",
            )
            lateral = self.fpn_lateral[i](encoder_features[i])
            fpn_features[i] = self.fpn_smooth[i](upsampled + lateral)

        # Full-resolution prediction from finest FPN level
        main_pred = self.out_conv(fpn_features[0])

        # Multi-scale predictions from levels at 1/4, 1/8, 1/16 resolution
        # encoder_features indices: 2 → 1/4, 3 → 1/8, 4 → 1/16
        scale_preds = []
        for head_idx, level_idx in enumerate([2, 3, 4]):
            pred = self.scale_heads[head_idx](fpn_features[level_idx])
            pred = F.interpolate(
                pred, size=input_size, mode="bilinear", align_corners=False
            )
            scale_preds.append(pred)

        # Weighted fusion of all scales
        weights = torch.softmax(self.scale_weights, dim=0)
        fused = (
            weights[0] * main_pred
            + weights[1] * scale_preds[0]
            + weights[2] * scale_preds[1]
            + weights[3] * scale_preds[2]
        )

        return torch.sigmoid(fused)


class RecognitionModel(nn.Module):
    """
    Text recognition model.

    This takes NCHW images of text lines as input and outputs a sequence of
    character predictions as a (W/4)xNxC tensor.

    The input images must be greyscale and have a fixed height of 64.

    The result is a sequence 1/4 the length of the input, where the `C` dim is
    the 1-based index of the character in the alphabet used to train the model.
    The value 0 is reserved for the blank character. The result sequence needs
    to be postprocessed with CTC decoding (eg. greedy or beam search) to recover
    the recognized character sequence.

    The model follows the general structure of CRNN [1], consisting of
    convolutional layers to extract features, followed by a bidirectional RNN to
    predict the character sequence.

    [1] https://arxiv.org/abs/1507.05717
    """

    def __init__(self, alphabet: str):
        """
        Construct the model.

        :param alphabet: Alphabet of characters that the model will recognize
        """

        super().__init__()

        n_classes = len(alphabet) + 1

        self.conv = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
                # Don't use biases for Conv2d when followed directly by batch norm,
                # per https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm.
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(
                128,
                128,
                kernel_size=(2, 2),
                padding=(1, 1),  # "same" padding
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=(4, 1)),
        )

        self.gru = nn.GRU(128, 256, bidirectional=True, num_layers=2)

        self.output = nn.Sequential(
            nn.Linear(512, n_classes),
            # nb. We use `LogSoftmax` here because `torch.nn.CTCLoss` expects log probs
            nn.LogSoftmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape

        x = self.conv(x)

        # Reshape from NCHW to WNCH
        x = torch.permute(x, (3, 0, 1, 2))

        # Combine last two dims to get WNx(CH)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))

        # Disable autocast here as PyTorch doesn't support GRU with bfloat16.
        with torch.autocast(x.device.type, enabled=False):
            x, _ = self.gru(x.float())

        return self.output(x)


class RecognitionModelV2Medium(nn.Module):
    """
    Larger CTC recognition model for better visual feature extraction.

    Same architecture as RecognitionModelV2 but with:
    - 2x wider CNN backbone (256 channels instead of 128)
    - Larger transformer encoder (d_model=384, nhead=6)
    - Better at distinguishing visually similar characters (w/v, 0/o)
    - Language-agnostic (CTC), works on all Latin-script languages
    """

    def __init__(self, alphabet: str):
        super().__init__()

        n_classes = len(alphabet) + 1
        cnn_channels = 256

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, cnn_channels, kernel_size=3, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=(1, 1), bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=(1, 1), bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.AvgPool2d(kernel_size=(4, 1)),
        )

        d_model = 384
        self.d_model = d_model

        self.project = nn.Linear(cnn_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=6,
            dim_feedforward=1536,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.output = nn.Sequential(
            nn.Linear(d_model, n_classes),
            nn.LogSoftmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = self.conv(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.project(x)

        seq_len = x.shape[1]
        pos_enc = positional_encoding(seq_len, self.d_model).to(x.device)
        x = x + pos_enc

        x = self.encoder(x)
        return self.output(x)


class RecognitionModelV2(nn.Module):
    """
    Text recognition model using Transformer encoder (replaces GRU).

    This takes NCHW images of text lines as input and outputs a sequence of
    character predictions as a [batch, W/4, C] tensor.

    The input images must be greyscale and have a fixed height of 64.

    The model follows a CNN + Transformer Encoder + CTC architecture:
    - CNN backbone (identical to RecognitionModel) extracts visual features
    - Transformer encoder replaces the bidirectional GRU for sequence modeling
    - CTC decoding recovers the character sequence

    Advantages over RecognitionModel (v1):
    - Transformers support bfloat16 (no autocast workaround needed)
    - Better parallelization during training
    - Global attention captures long-range dependencies
    """

    def __init__(self, alphabet: str):
        """
        Construct the model.

        :param alphabet: Alphabet of characters that the model will recognize
        """
        super().__init__()

        n_classes = len(alphabet) + 1

        # CNN backbone identical to RecognitionModel (downsample factor = 4)
        self.conv = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(
                128,
                128,
                kernel_size=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=(4, 1)),
        )

        d_model = 256

        # Projection from CNN features to transformer dimension
        self.project = nn.Linear(128, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.d_model = d_model

        self.output = nn.Sequential(
            nn.Linear(d_model, n_classes),
            nn.LogSoftmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape

        x = self.conv(x)

        # Reshape from NCHW to [N, W, C*H] (H=1 after CNN so C*H = 128)
        x = x.squeeze(2)  # [N, 128, W']
        x = x.permute(0, 2, 1)  # [N, W', 128]

        # Project to d_model
        x = self.project(x)  # [N, W', 256]

        # Add sinusoidal positional encoding
        seq_len = x.shape[1]
        pos_enc = positional_encoding(seq_len, self.d_model).to(x.device)
        x = x + pos_enc

        # Transformer encoder
        x = self.encoder(x)  # [N, W', 256]

        # Output: [batch, seq, n_classes]
        return self.output(x)


class RecognitionModelV3(nn.Module):
    """
    Text recognition model with autoregressive Transformer decoder.

    Architecture: CNN backbone + Transformer Encoder + Transformer Decoder.
    Unlike v1 (CTC+GRU) and v2 (CTC+Transformer Encoder), this model uses
    cross-attention between the decoder and encoder features, allowing each
    predicted character to attend to the full visual context and previous
    predictions. This resolves CTC's independence assumption that causes
    confusions between visually similar characters (w/v, 0/o, d/a).

    Special tokens:
    - 0: PAD
    - 1: SOS (start of sequence)
    - 2: EOS (end of sequence)
    - 3+: alphabet characters (alphabet[i] = token i+3)
    """

    PAD_TOKEN = 0
    SOS_TOKEN = 1
    EOS_TOKEN = 2
    CHAR_OFFSET = 3

    def __init__(self, alphabet: str, max_seq_len: int = 150):
        super().__init__()

        self.alphabet = alphabet
        self.max_seq_len = max_seq_len
        vocab_size = len(alphabet) + self.CHAR_OFFSET  # PAD + SOS + EOS + chars

        # CNN backbone (same as v1/v2)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(128, 128, kernel_size=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=(4, 1)),
        )

        d_model = 256
        self.d_model = d_model

        # Project CNN features to d_model
        self.project = nn.Linear(128, d_model)

        # Transformer encoder (same as v2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Decoder components
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run CNN + encoder on image. Returns encoder memory."""
        x = self.conv(x)
        x = x.squeeze(2)  # [N, 128, W']
        x = x.permute(0, 2, 1)  # [N, W', 128]
        x = self.project(x)  # [N, W', 256]

        seq_len = x.shape[1]
        pos_enc = positional_encoding(seq_len, self.d_model).to(x.device)
        x = x + pos_enc

        return self.encoder(x)  # [N, W', 256]

    def decode(
        self, memory: torch.Tensor, tgt_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Run decoder on target tokens with encoder memory.

        :param memory: [N, src_len, d_model] encoder output
        :param tgt_tokens: [N, tgt_len] token indices
        :return: [N, tgt_len, vocab_size] logits
        """
        tgt_len = tgt_tokens.shape[1]

        # Token embedding + positional encoding
        tgt = self.token_embedding(tgt_tokens)  # [N, tgt_len, d_model]
        pos_enc = positional_encoding(tgt_len, self.d_model).to(tgt.device)
        tgt = tgt + pos_enc

        # Causal mask: prevent attending to future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt.device
        )

        out = self.decoder(tgt, memory, tgt_mask=causal_mask)  # [N, tgt_len, d_model]
        return self.output_proj(out)  # [N, tgt_len, vocab_size]

    def forward(
        self, x: torch.Tensor, tgt_tokens: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass.

        During training, pass tgt_tokens for teacher forcing.
        During inference (tgt_tokens=None), decode autoregressively.
        """
        memory = self.encode(x)

        if tgt_tokens is not None:
            # Training: teacher forcing
            return self.decode(memory, tgt_tokens)

        # Inference: autoregressive decoding
        return self.inference(memory)

    @torch.no_grad()
    def inference(self, memory: torch.Tensor) -> torch.Tensor:
        """Autoregressive greedy decoding."""
        batch_size = memory.shape[0]
        device = memory.device

        # Start with SOS token
        tokens = torch.full(
            (batch_size, 1), self.SOS_TOKEN, dtype=torch.long, device=device
        )

        for _ in range(self.max_seq_len):
            logits = self.decode(memory, tokens)  # [N, seq, vocab]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [N, 1]
            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop if all sequences have produced EOS
            if (next_token == self.EOS_TOKEN).all():
                break

        return tokens  # [N, seq_len] token indices

    def encode_target(self, text: str) -> torch.Tensor:
        """Convert text string to target token sequence (with SOS/EOS)."""
        alphabet_list = list(self.alphabet)
        tokens = [self.SOS_TOKEN]
        for ch in text:
            try:
                idx = alphabet_list.index(ch)
                tokens.append(idx + self.CHAR_OFFSET)
            except ValueError:
                # Unknown char — skip
                pass
        tokens.append(self.EOS_TOKEN)
        return torch.tensor(tokens, dtype=torch.long)

    def decode_tokens(self, tokens: torch.Tensor | list[int]) -> str:
        """Convert token indices back to text string."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        alphabet_list = list(self.alphabet)
        chars = []
        for t in tokens:
            if t == self.EOS_TOKEN:
                break
            if t >= self.CHAR_OFFSET:
                idx = t - self.CHAR_OFFSET
                if idx < len(alphabet_list):
                    chars.append(alphabet_list[idx])
        return "".join(chars)


class RecognitionModelV3Export(nn.Module):
    """ONNX-export wrapper for RecognitionModelV3.

    Exports two entry points:
    - encode(image) -> memory
    - decode(memory, tokens) -> logits

    For ONNX we export them as a single model with the encoder part.
    The autoregressive loop happens in the Rust runtime.
    """

    def __init__(self, conv, project, d_model, encoder_layers, decoder_layers, token_embedding, output_proj):
        super().__init__()
        self.conv = conv
        self.project = project
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.token_embedding = token_embedding
        self.output_proj = output_proj

    @staticmethod
    def _copy_encoder_weights(src, dst):
        """Copy weights from a PyTorch TransformerEncoderLayer to a _ManualEncoderLayer."""
        dst.self_attn.qkv_proj.weight.data = src.self_attn.in_proj_weight.data
        dst.self_attn.qkv_proj.bias.data = src.self_attn.in_proj_bias.data
        dst.self_attn.out_proj.weight.data = src.self_attn.out_proj.weight.data
        dst.self_attn.out_proj.bias.data = src.self_attn.out_proj.bias.data
        dst.linear1.weight.data = src.linear1.weight.data
        dst.linear1.bias.data = src.linear1.bias.data
        dst.linear2.weight.data = src.linear2.weight.data
        dst.linear2.bias.data = src.linear2.bias.data
        dst.norm1.weight.data = src.norm1.weight.data
        dst.norm1.bias.data = src.norm1.bias.data
        dst.norm2.weight.data = src.norm2.weight.data
        dst.norm2.bias.data = src.norm2.bias.data

    @staticmethod
    def _copy_decoder_weights(src, dst):
        """Copy weights from a PyTorch TransformerDecoderLayer to a _ManualDecoderLayer."""
        # Self-attention (in decoder, self_attn uses in_proj_weight for qkv)
        dst.self_attn.qkv_proj.weight.data = src.self_attn.in_proj_weight.data
        dst.self_attn.qkv_proj.bias.data = src.self_attn.in_proj_bias.data
        dst.self_attn.out_proj.weight.data = src.self_attn.out_proj.weight.data
        dst.self_attn.out_proj.bias.data = src.self_attn.out_proj.bias.data

        # Cross-attention (multihead_attn in PyTorch decoder)
        # PyTorch stores q, k, v projections in in_proj_weight [3*d, d]
        d = src.multihead_attn.embed_dim
        in_proj_w = src.multihead_attn.in_proj_weight.data
        in_proj_b = src.multihead_attn.in_proj_bias.data
        dst.cross_attn.q_proj.weight.data = in_proj_w[:d]
        dst.cross_attn.q_proj.bias.data = in_proj_b[:d]
        dst.cross_attn.k_proj.weight.data = in_proj_w[d:2*d]
        dst.cross_attn.k_proj.bias.data = in_proj_b[d:2*d]
        dst.cross_attn.v_proj.weight.data = in_proj_w[2*d:]
        dst.cross_attn.v_proj.bias.data = in_proj_b[2*d:]
        dst.cross_attn.out_proj.weight.data = src.multihead_attn.out_proj.weight.data
        dst.cross_attn.out_proj.bias.data = src.multihead_attn.out_proj.bias.data

        # FFN
        dst.linear1.weight.data = src.linear1.weight.data
        dst.linear1.bias.data = src.linear1.bias.data
        dst.linear2.weight.data = src.linear2.weight.data
        dst.linear2.bias.data = src.linear2.bias.data

        # Norms (decoder has 3: norm1=self_attn, norm2=cross_attn, norm3=ffn)
        dst.norm1.weight.data = src.norm1.weight.data
        dst.norm1.bias.data = src.norm1.bias.data
        dst.norm2.weight.data = src.norm2.weight.data
        dst.norm2.bias.data = src.norm2.bias.data
        dst.norm3.weight.data = src.norm3.weight.data
        dst.norm3.bias.data = src.norm3.bias.data

    @staticmethod
    def from_trained(model: "RecognitionModelV3") -> "RecognitionModelV3Export":
        """Build export model from trained model."""
        d_model = model.d_model
        nhead = model.encoder.layers[0].self_attn.num_heads
        enc_ff = model.encoder.layers[0].linear1.out_features
        dec_ff = model.decoder.layers[0].linear1.out_features

        # Build manual encoder layers
        n_enc = len(model.encoder.layers)
        encoder_layers = nn.ModuleList(
            [_ManualEncoderLayer(d_model, nhead, enc_ff) for _ in range(n_enc)]
        )
        for i in range(n_enc):
            RecognitionModelV3Export._copy_encoder_weights(
                model.encoder.layers[i], encoder_layers[i]
            )

        # Build manual decoder layers
        n_dec = len(model.decoder.layers)
        decoder_layers = nn.ModuleList(
            [_ManualDecoderLayer(d_model, nhead, dec_ff) for _ in range(n_dec)]
        )
        for i in range(n_dec):
            RecognitionModelV3Export._copy_decoder_weights(
                model.decoder.layers[i], decoder_layers[i]
            )

        return RecognitionModelV3Export(
            conv=model.conv,
            project=model.project,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            token_embedding=model.token_embedding,
            output_proj=model.output_proj,
        )

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        # Encode
        x = self.conv(image)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.project(x)

        seq_len = x.shape[1]
        pos_enc = positional_encoding(seq_len, self.d_model).to(x.device)
        x = x + pos_enc

        # Manual encoder [batch, seq, d] -> [seq, batch, d]
        x = x.permute(1, 0, 2)
        for layer in self.encoder_layers:
            x = layer(x)
        memory = x  # keep as [seq, batch, d]

        # Decode
        tgt = self.token_embedding(tokens)  # [batch, tgt_len, d]
        tgt_len = tgt.shape[1]
        tgt_pos = positional_encoding(tgt_len, self.d_model).to(tgt.device)
        tgt = tgt + tgt_pos

        # [batch, tgt_len, d] -> [tgt_len, batch, d]
        tgt = tgt.permute(1, 0, 2)
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)
        # [tgt_len, batch, d] -> [batch, tgt_len, d]
        tgt = tgt.permute(1, 0, 2)

        return self.output_proj(tgt)


class _ManualMultiheadAttention(nn.Module):
    """Multi-head attention using simple ops for clean ONNX export.

    ``nn.MultiheadAttention`` relies on ``scaled_dot_product_attention`` which
    PyTorch decomposes into Reshape ops with constant-folded shapes during ONNX
    export. Those dynamic Reshapes are not supported by RTen.  This module
    produces the exact same result but with ONNX-friendly ops only.
    """

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.d_model = d_model
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        seq, batch, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(seq, batch * self.nhead, self.d_k).transpose(0, 1)
        k = k.reshape(seq, batch * self.nhead, self.d_k).transpose(0, 1)
        v = v.reshape(seq, batch * self.nhead, self.d_k).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)

        if causal:
            mask = torch.triu(torch.ones(seq, seq, device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(0, 1).reshape(seq, batch, self.d_model)
        return self.out_proj(out)

    def forward_causal(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, causal=True)


class _ManualCrossAttention(nn.Module):
    """Cross-attention using simple ops for clean ONNX export.

    Query comes from the decoder, key/value from the encoder memory.
    """

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        :param query: [seq_q, batch, d_model] from decoder
        :param memory: [seq_m, batch, d_model] from encoder
        """
        seq_q, batch, _ = query.shape
        seq_m = memory.shape[0]

        q = self.q_proj(query)
        k = self.k_proj(memory)
        v = self.v_proj(memory)

        q = q.reshape(seq_q, batch * self.nhead, self.d_k).transpose(0, 1)
        k = k.reshape(seq_m, batch * self.nhead, self.d_k).transpose(0, 1)
        v = v.reshape(seq_m, batch * self.nhead, self.d_k).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(0, 1).reshape(seq_q, batch, self.d_model)
        return self.out_proj(out)


class _ManualDecoderLayer(nn.Module):
    """Transformer decoder layer with manual attention for ONNX export."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.self_attn = _ManualMultiheadAttention(d_model, nhead)
        self.cross_attn = _ManualCrossAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention with causal mask
        x2 = self.self_attn.forward_causal(tgt)
        tgt = self.norm1(tgt + x2)
        # Cross-attention
        x2 = self.cross_attn(tgt, memory)
        tgt = self.norm2(tgt + x2)
        # FFN
        x2 = self.linear2(torch.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + x2)
        return tgt


class _ManualEncoderLayer(nn.Module):
    """Transformer encoder layer with manual attention for ONNX export."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.self_attn = _ManualMultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.self_attn(x)
        x = self.norm1(x + x2)
        x2 = self.linear2(torch.relu(self.linear1(x)))
        x = self.norm2(x + x2)
        return x


class RecognitionModelV2Export(nn.Module):
    """ONNX-export wrapper for :class:`RecognitionModelV2`.

    This replaces ``nn.TransformerEncoderLayer`` (which uses
    ``scaled_dot_product_attention``) with a manual implementation that
    produces an ONNX graph compatible with RTen.  Weights are copied
    from a trained ``RecognitionModelV2`` — the numerical output is
    identical (< 1e-5 difference).

    Usage::

        model = RecognitionModelV2(alphabet=...).load_state_dict(...)
        export_model = RecognitionModelV2Export.from_trained(model)
        torch.onnx.export(export_model, ...)
    """

    def __init__(self, conv, project, d_model, output, n_layers: int = 4, nhead: int = 4, dim_feedforward: int = 1024):
        super().__init__()
        self.conv = conv
        self.project = project
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [_ManualEncoderLayer(d_model, nhead, dim_feedforward) for _ in range(n_layers)]
        )
        self.output = output

    @staticmethod
    def from_trained(model) -> "RecognitionModelV2Export":
        """Build an export model by copying weights from a trained model (v2 or v2m)."""
        n_layers = len(model.encoder.layers)
        src_layer = model.encoder.layers[0]
        nhead = src_layer.self_attn.num_heads
        dim_feedforward = src_layer.linear1.out_features
        export_model = RecognitionModelV2Export(
            conv=model.conv,
            project=model.project,
            d_model=model.d_model,
            output=model.output,
            n_layers=n_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )

        for i in range(n_layers):
            src = model.encoder.layers[i]
            dst = export_model.layers[i]

            # Copy multi-head attention weights
            dst.self_attn.qkv_proj.weight.data = src.self_attn.in_proj_weight.data
            dst.self_attn.qkv_proj.bias.data = src.self_attn.in_proj_bias.data
            dst.self_attn.out_proj.weight.data = src.self_attn.out_proj.weight.data
            dst.self_attn.out_proj.bias.data = src.self_attn.out_proj.bias.data

            # Copy FFN weights
            dst.linear1.weight.data = src.linear1.weight.data
            dst.linear1.bias.data = src.linear1.bias.data
            dst.linear2.weight.data = src.linear2.weight.data
            dst.linear2.bias.data = src.linear2.bias.data

            # Copy layer norms
            dst.norm1.weight.data = src.norm1.weight.data
            dst.norm1.bias.data = src.norm1.bias.data
            dst.norm2.weight.data = src.norm2.weight.data
            dst.norm2.bias.data = src.norm2.bias.data

        return export_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.project(x)

        seq_len = x.shape[1]
        pos_enc = positional_encoding(seq_len, self.d_model).to(x.device)
        x = x + pos_enc

        # [batch, seq, d] → [seq, batch, d] for the manual encoder layers
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x)
        # [seq, batch, d] → [batch, seq, d]
        x = x.permute(1, 0, 2)

        return self.output(x)


def positional_encoding(length: torch.Tensor, depth: int) -> torch.Tensor:
    """
    Generate a tensor of sinusoidal position encodings.

    The returned tensor has shape `(length, depth)`. If `depth` is odd, it will
    be rounded down to the nearest even number.

    This is a slightly modified version of the positional encodings in the
    original transformer paper, based on
    https://www.tensorflow.org/text/tutorials/transformer and
    https://jalammar.github.io/illustrated-transformer/.

    :param length: Number of positions to generate encodings for
    :param depth: The size of the encoding vector for each position
    """
    depth = depth // 2

    # (length, 1)
    positions = torch.arange(length).unsqueeze(-1)  # type: ignore
    depths = torch.arange(depth).unsqueeze(0) / depth  # (1, depth)

    angle_rates = 1 / (10_000**depths)
    angle_rads = positions * angle_rates  # (length, depth)

    return torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], dim=-1)


def encode_bbox_positions(boxes: torch.Tensor, size: int) -> torch.Tensor:
    """
    Convert bounding box positions to positional encodings.

    :param boxes: (N, W, D) tensor of bounding box coordinates, where D = 4.
    :param size: Size of encoding for each coordinate
    :return: (N, W, D * size) tensor of encodings
    """
    N, W, D = boxes.shape
    # assert D == 4  # Should be [left, top, right, bottom]

    int_boxes = boxes.round().int()
    max_coord = int_boxes.max()

    encodings = positional_encoding(max_coord + 1, size).to(
        boxes.device
    )  # (max_coord, size)
    encoded = encodings[int_boxes]  # (N, W, D, size)
    encoded = encoded.reshape((N, W, D * size))

    return encoded


class SinPositionalEncoding(nn.Module):
    """
    Converting coordinates of bounding boxes to sinusoidal position encodings.

    See `encode_bbox_positions`.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, boxes):
        """
        :param boxes: (N, W, D) tensor of bounding box coordinates
        """
        N, W, D = boxes.shape
        return encode_bbox_positions(boxes, self.d_model // D)


class LayoutModel(nn.Module):
    """
    Text layout analysis model.

    Inputs have shape `[N, W, D]` where N is the batch size, W is the word
    index, D is the word feature index.

    Outputs have shape `[N, W, C]` where C is a vector of either logits or
    probabilities for different word attributes: `[line_start, line_end]`.
    """

    embed: nn.Module

    def __init__(
        self, return_probs=False, pos_embedding: Literal["mlp", "sin"] = "sin"
    ):
        """

        :param return_probs: If true, the model returns probabilities, otherwise
            it returns logits which can be converted to probabilities using
            sigmoid.
        """
        super().__init__()

        n_features = 4
        d_model = 256
        d_feedforward = d_model * 4
        n_classes = 2
        n_layers = 6
        n_heads = max(d_model // 64, 1)

        self.d_embed = d_model
        self.return_probs = return_probs

        match pos_embedding:
            case "mlp":
                self.embed = nn.Sequential(
                    nn.Linear(n_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, d_model),
                    nn.ReLU(),
                )
            case "sin":
                self.embed = SinPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_feedforward
        )
        self.encode = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classify = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Tensor of (N, W, D) features for word bounding boxes
        :return: Tensor of (N, W, C) logits or probabilities for different word
            attributes.
        """
        x = self.embed(x)
        x = self.encode(x)
        x = self.classify(x)

        if self.return_probs:
            return x.sigmoid()
        else:
            return x

"""
Training script for RecognitionModelV3 (autoregressive decoder).

Unlike train_rec.py which uses CTC loss, this uses cross-entropy with
teacher forcing for the autoregressive Transformer decoder.
"""

from argparse import ArgumentParser, BooleanOptionalAction
import math
import os

from pylev import levenshtein
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from .datasets.hiertext import DEFAULT_ALPHABET, HierTextRecognition
from .datasets.textocr import TextOCRRecognition
from .datasets.combined import CombinedDataset
from .datasets import text_recognition_data_augmentations
from .datasets.util import normalize_text
from .models import RecognitionModelV3, RecognitionModelV3Export
from .train_detection import load_checkpoint, save_checkpoint


def encode_text_v3(text: str, alphabet: str, max_len: int) -> tuple[torch.Tensor, int]:
    """
    Encode text for v3 autoregressive training.

    Returns (tokens, length) where tokens includes SOS at start and EOS at end,
    padded to max_len.
    """
    SOS = RecognitionModelV3.SOS_TOKEN
    EOS = RecognitionModelV3.EOS_TOKEN
    PAD = RecognitionModelV3.PAD_TOKEN
    OFFSET = RecognitionModelV3.CHAR_OFFSET

    text = normalize_text(text)
    alphabet_list = list(alphabet)

    tokens = [SOS]
    for ch in text:
        try:
            idx = alphabet_list.index(ch)
            tokens.append(idx + OFFSET)
        except ValueError:
            pass  # skip unknown chars
    tokens.append(EOS)

    length = len(tokens)
    # Pad to max_len
    tokens = tokens + [PAD] * (max_len - length)
    return torch.tensor(tokens[:max_len], dtype=torch.long), min(length, max_len)


def collate_samples_v3(samples: list[dict]) -> dict:
    """Collate samples for v3 training."""

    def image_width(sample: dict) -> int:
        return sample["image"].shape[-1]

    # Round up image width to reduce tensor size variation
    img_width_step = 256
    max_img_width = max(image_width(s) for s in samples)
    max_img_width = ((max_img_width + img_width_step - 1) // img_width_step) * img_width_step

    # Pre-compute text lengths (SOS + chars + EOS)
    for sample in samples:
        text = sample.get("text", "")
        sample["text_len_v3"] = len(normalize_text(text)) + 2  # +SOS+EOS

    max_text_len = max(s["text_len_v3"] for s in samples)
    max_text_len = max(max_text_len, 4)

    for sample in samples:
        # Pad image
        w = image_width(sample)
        sample["image_width"] = w
        sample["image"] = F.pad(
            sample["image"],
            [0, max_img_width - w],
            mode="constant",
            value=0.0,
        )

        # Re-encode text with uniform max_len
        text = sample.get("text", "")
        tokens, length = encode_text_v3(text, DEFAULT_ALPHABET, max_text_len)
        sample["tgt_tokens"] = tokens
        sample["tgt_length"] = length

        # Remove fields that can't be collated uniformly
        sample.pop("text_seq", None)
        sample.pop("text", None)
        sample.pop("text_len_v3", None)
        sample.pop("image_id", None)

    return default_collate(samples)


class V3AccuracyStats:
    """Accuracy stats for autoregressive model."""

    def __init__(self):
        self.total_chars = 0
        self.char_errors = 0

    def update(self, pred_texts: list[str], target_texts: list[str]):
        for pred, target in zip(pred_texts, target_texts):
            self.total_chars += len(target)
            self.char_errors += levenshtein(target, pred)

    def char_error_rate(self) -> float:
        if self.total_chars == 0:
            return 0.0
        return self.char_errors / self.total_chars


def decode_tokens_batch(tokens: torch.Tensor, model: RecognitionModelV3) -> list[str]:
    """Decode a batch of token sequences to strings."""
    results = []
    for i in range(tokens.shape[0]):
        results.append(model.decode_tokens(tokens[i]))
    return results


def decode_target_batch(tokens: torch.Tensor, lengths: list[int], model: RecognitionModelV3) -> list[str]:
    """Decode target tokens (skip SOS, stop at EOS)."""
    results = []
    for i in range(tokens.shape[0]):
        text = model.decode_tokens(tokens[i, 1:lengths[i]])  # skip SOS
        results.append(text)
    return results


def train_epoch(
    epoch: int,
    device: torch.device,
    dataloader: DataLoader,
    model: RecognitionModelV3,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, V3AccuracyStats]:
    model.train()
    train_iter = tqdm(dataloader)
    train_iter.set_description(f"Training (epoch {epoch})")
    total_loss = 0.0
    stats = V3AccuracyStats()

    for batch_idx, batch in enumerate(train_iter):
        img = batch["image"].to(device)
        tgt_tokens = batch["tgt_tokens"].to(device)
        tgt_lengths = batch["tgt_length"]

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # Teacher forcing: input is tokens[:-1], target is tokens[1:]
            decoder_input = tgt_tokens[:, :-1]
            decoder_target = tgt_tokens[:, 1:]

            logits = model(img, decoder_input)  # [N, seq-1, vocab]

            # Cross-entropy loss, ignoring PAD tokens
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                decoder_target.reshape(-1),
                ignore_index=RecognitionModelV3.PAD_TOKEN,
            )

        if math.isnan(loss.item()):
            raise Exception("NaN loss encountered")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()
        total_loss += loss.item()

        # Compute accuracy on greedy predictions
        with torch.no_grad():
            pred_tokens = logits.argmax(dim=-1)  # [N, seq-1]
            pred_texts = []
            target_texts = []
            for i in range(pred_tokens.shape[0]):
                pred_texts.append(model.decode_tokens(pred_tokens[i]))
                target_texts.append(model.decode_tokens(decoder_target[i]))
            stats.update(pred_texts, target_texts)

        # Preview first batch
        if batch_idx == 0:
            for i in range(min(5, len(pred_texts))):
                print(f'  Train: "{pred_texts[i]}" <- "{target_texts[i]}"')

    train_iter.clear()
    return total_loss / len(dataloader), stats


@torch.no_grad()
def validate(
    device: torch.device,
    dataloader: DataLoader,
    model: RecognitionModelV3,
) -> tuple[float, V3AccuracyStats]:
    model.eval()
    val_iter = tqdm(dataloader)
    val_iter.set_description("Validating")
    total_loss = 0.0
    stats = V3AccuracyStats()

    for batch_idx, batch in enumerate(val_iter):
        img = batch["image"].to(device)
        tgt_tokens = batch["tgt_tokens"].to(device)
        tgt_lengths = batch["tgt_length"]

        decoder_input = tgt_tokens[:, :-1]
        decoder_target = tgt_tokens[:, 1:]

        logits = model(img, decoder_input)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            decoder_target.reshape(-1),
            ignore_index=RecognitionModelV3.PAD_TOKEN,
        )
        total_loss += loss.item()

        # Greedy decode from logits (teacher-forced)
        pred_tokens = logits.argmax(dim=-1)
        pred_texts = []
        target_texts = []
        for i in range(pred_tokens.shape[0]):
            pred_texts.append(model.decode_tokens(pred_tokens[i]))
            target_texts.append(model.decode_tokens(decoder_target[i]))
        stats.update(pred_texts, target_texts)

        if batch_idx == 0:
            for i in range(min(5, len(pred_texts))):
                print(f'  Val: "{pred_texts[i]}" <- "{target_texts[i]}"')

    val_iter.clear()
    return total_loss / len(dataloader), stats


def main():
    parser = ArgumentParser(description="Train text recognition model v3 (autoregressive).")
    parser.add_argument("dataset_type", type=str, choices=["hiertext", "textocr", "combined"])
    parser.add_argument("data_dir")
    parser.add_argument("--augment", default=True, action=BooleanOptionalAction)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--textocr-dir", type=str)
    parser.add_argument("--export", type=str, help="Export model to ONNX")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1234)

    max_images = args.max_images
    validation_max_images = max(10, int(max_images * 0.1)) if max_images else None

    augmentations = text_recognition_data_augmentations() if args.augment else None

    # Load datasets
    if args.dataset_type == "combined":
        if not args.textocr_dir:
            raise Exception("--textocr-dir required for combined dataset")
        hiertext_train = HierTextRecognition(args.data_dir, train=True, max_images=max_images, transform=augmentations)
        textocr_train = TextOCRRecognition(args.textocr_dir, train=True, max_images=max_images, transform=augmentations)
        train_dataset = CombinedDataset(datasets=[hiertext_train, textocr_train], ratios=[0.8, 0.2])
        val_dataset = HierTextRecognition(args.data_dir, train=False, max_images=validation_max_images)
    elif args.dataset_type == "hiertext":
        train_dataset = HierTextRecognition(args.data_dir, train=True, max_images=max_images, transform=augmentations)
        val_dataset = HierTextRecognition(args.data_dir, train=False, max_images=validation_max_images)
    elif args.dataset_type == "textocr":
        train_dataset = TextOCRRecognition(args.textocr_dir or args.data_dir, train=True, max_images=max_images, transform=augmentations)
        val_dataset = TextOCRRecognition(args.textocr_dir or args.data_dir, train=False, max_images=validation_max_images)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_samples_v3, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_samples_v3, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecognitionModelV3(alphabet=DEFAULT_ALPHABET).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model param count {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.max_epochs - warmup_epochs)),
        ],
        milestones=[warmup_epochs],
    )

    epoch = 0
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer, device)
        epoch = checkpoint["epoch"]

    if args.export:
        model.eval()
        export_model = RecognitionModelV3Export.from_trained(model).to(device)
        export_model.eval()

        # Dummy inputs
        dummy_img = torch.randn(1, 1, 64, 256).to(device)
        dummy_tokens = torch.tensor([[1, 5, 10]], dtype=torch.long).to(device)

        torch.onnx.export(
            export_model,
            (dummy_img, dummy_tokens),
            args.export,
            input_names=["image", "tokens"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch", 3: "width"},
                "tokens": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            opset_version=17,
        )

        import onnx
        onnx_model = onnx.load(args.export)
        onnx_model.metadata_props.append(
            onnx.StringStringEntryProto(key="alphabet", value=DEFAULT_ALPHABET)
        )
        onnx_model.metadata_props.append(
            onnx.StringStringEntryProto(key="decoder", value="autoregressive")
        )
        onnx.save(onnx_model, args.export)
        print(f"Exported v3 model with alphabet ({len(DEFAULT_ALPHABET)} chars)")
        return

    if args.validate_only:
        val_loss, val_stats = validate(device, val_loader, model)
        print(f"Val loss {val_loss:.4f} CER {val_stats.char_error_rate():.4f}")
        return

    # Training loop
    best_val_loss = float("inf")
    patience = 10
    no_improve = 0

    while epoch < args.max_epochs:
        train_loss, train_stats = train_epoch(epoch, device, train_loader, model, optimizer)
        print(f"Epoch {epoch} train loss {train_loss:.4f} CER {train_stats.char_error_rate():.4f}")

        val_loss, val_stats = validate(device, val_loader, model)
        print(f"Epoch {epoch} val loss {val_loss:.4f} CER {val_stats.char_error_rate():.4f}")

        scheduler.step()
        print(f"LR {scheduler.get_last_lr()}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint("text-rec-v3-checkpoint.pt", model, optimizer, epoch=epoch)
            print(f"  Saved checkpoint (best val loss: {best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

        epoch += 1


if __name__ == "__main__":
    main()

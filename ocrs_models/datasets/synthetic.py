"""
Synthetic text recognition dataset.

Generates text line images on-the-fly using PIL/Pillow with random fonts, text,
and augmentations. Useful for pre-training or augmenting real-world datasets
when annotated data is scarce.

Each item matches the output format of HierTextRecognition: a dict with
"image" (CHW float tensor, values in [-0.5, 0.5]), "text_seq" (encoded label
tensor), "text" (raw string), and "image_id" (str).
"""

import math
import os
import platform
import random
from typing import Optional, cast

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torchvision.transforms.functional import resize

from .util import SizedDataset, encode_text, normalize_text, transform_image

# ---------------------------------------------------------------------------
# Built-in word lists for Latin-script languages. Each list contains common
# words, numbers, punctuation fragments, and a handful of special characters
# so that the generator can produce realistic text lines.
# ---------------------------------------------------------------------------

_COMMON_NUMBERS = [
    "0", "1", "2", "3", "10", "15", "20", "42", "50", "99", "100", "200",
    "500", "1000", "2024", "2025", "2026", "3.14", "0.5", "12.99",
    "$9.99", "$19.95", "$100", "50%", "25%", "100%", "#1", "#42",
    "1st", "2nd", "3rd", "4th", "5th", "10th",
]

_PUNCTUATION_FRAGMENTS = [
    ".", ",", "!", "?", ":", ";", "-", "(", ")", '"', "'",
    "...", "--", "—", "/", "&", "@", "+", "=", "*",
    "«", "»", "¿", "¡",
]

_SPECIAL_FRAGMENTS = [
    "©2025", "®", "™", "§1", "§2", "€50", "€100", "$25",
    "½", "¼", "¾", "±5", "°C", "°F", "²", "³",
]

_WORD_LISTS: dict[str, list[str]] = {
    "en": [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "world", "very", "company", "market",
        "price", "information", "system", "number", "part", "place",
        "while", "found", "each", "under", "between", "city", "never",
        "every", "start", "might", "story", "children", "always",
        "example", "paper", "often", "together", "running", "important",
        "without", "small", "however", "before", "another", "question",
        "Monday", "Tuesday", "January", "February", "March", "April",
        "Hello", "Goodbye", "Thank", "Please", "Welcome", "Street",
        "Avenue", "Building", "Department", "University", "School",
    ],
    "fr": [
        "le", "de", "un", "être", "et", "à", "il", "avoir", "ne", "je",
        "son", "que", "se", "qui", "ce", "dans", "en", "du", "elle", "au",
        "pas", "pour", "plus", "par", "sur", "faire", "avec", "tout",
        "dire", "comme", "mais", "nous", "voir", "aussi", "bien", "où",
        "très", "même", "autre", "donner", "premier", "notre", "après",
        "monde", "année", "jour", "homme", "temps", "vie", "main",
        "encore", "moment", "petit", "rien", "alors", "pays", "entre",
        "sans", "tête", "grand", "chez", "politique", "état", "avant",
        "être", "depuis", "maison", "déjà", "toujours", "français",
        "bonjour", "merci", "société", "différent", "général", "école",
        "président", "gouvernement", "développement", "résultat",
        "problème", "système", "marché", "qualité", "aujourd'hui",
        "République", "Liberté", "Égalité", "Fraternité",
    ],
    "de": [
        "der", "die", "und", "in", "den", "von", "zu", "das", "mit",
        "sich", "des", "auf", "für", "ist", "im", "dem", "nicht", "ein",
        "eine", "als", "auch", "es", "an", "werden", "aus", "er", "hat",
        "dass", "sie", "nach", "wird", "bei", "einer", "um", "noch",
        "wie", "einem", "über", "so", "zum", "kann", "diese", "nur",
        "oder", "aber", "vor", "bis", "mehr", "durch", "man", "dann",
        "soll", "schon", "wenn", "was", "vom", "gut", "Zeit", "Stadt",
        "Straße", "groß", "zwischen", "Gesellschaft", "Universität",
        "können", "müssen", "während", "öffentlich", "möglich",
        "Geschäft", "Gebäude", "Größe", "Düsseldorf", "München",
        "Zürich", "Österreich", "Übung", "geöffnet", "geschlossen",
    ],
    "es": [
        "de", "la", "que", "el", "en", "y", "a", "los", "se", "del",
        "las", "un", "por", "con", "no", "una", "su", "para", "es", "al",
        "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este",
        "si", "me", "sin", "sobre", "este", "entre", "cuando", "muy",
        "ser", "también", "otro", "hasta", "desde", "donde", "tiempo",
        "año", "así", "después", "puede", "gobierno", "día", "vida",
        "parte", "país", "mundo", "caso", "hacer", "mejor", "hombre",
        "ciudad", "trabajo", "mujer", "niño", "número", "señor",
        "España", "información", "educación", "comunicación",
        "situación", "económico", "político", "público", "español",
    ],
    "pt": [
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
        "com", "não", "uma", "os", "no", "se", "na", "por", "mais",
        "as", "dos", "como", "mas", "ao", "ele", "das", "à", "seu",
        "sua", "ou", "quando", "muito", "nos", "já", "eu", "também",
        "só", "pelo", "pela", "até", "isso", "ela", "entre", "depois",
        "sem", "mesmo", "aos", "seus", "quem", "nas", "esse", "todos",
        "governo", "país", "cidade", "vida", "tempo", "dia", "mundo",
        "casa", "trabalho", "estado", "história", "Brasil", "São",
        "Paulo", "ação", "educação", "informação", "situação",
        "comunicação", "número", "público", "português",
    ],
    "it": [
        "di", "e", "il", "la", "che", "in", "un", "a", "per", "è",
        "una", "del", "non", "si", "con", "lo", "le", "da", "come",
        "sono", "io", "ci", "questo", "ha", "ma", "suo", "al", "dei",
        "più", "anche", "era", "molto", "dopo", "essere", "nel",
        "degli", "stato", "tutto", "tra", "fatto", "quando", "vita",
        "ancora", "altro", "ogni", "tanto", "prima", "parte", "tempo",
        "grande", "anno", "dove", "mondo", "città", "casa", "lavoro",
        "giorno", "uomo", "paese", "modo", "storia", "governo",
        "Italia", "società", "politica", "qualità", "università",
        "informazione", "perché", "così", "già",
    ],
    "nl": [
        "de", "van", "een", "het", "en", "in", "is", "dat", "op", "te",
        "zijn", "voor", "met", "die", "niet", "hij", "maar", "er", "aan",
        "ook", "als", "om", "dan", "nog", "bij", "dit", "uit", "al",
        "was", "wat", "worden", "kan", "naar", "over", "heeft", "meer",
        "men", "door", "veel", "geen", "moet", "wel", "tot", "zo", "na",
        "haar", "jaar", "tijd", "stad", "werk", "land", "huis", "dag",
        "leven", "mensen", "wereld", "groot", "politiek", "regering",
        "maatschappij", "informatie", "universiteit", "kwaliteit",
        "Nederland", "Amsterdam", "straat", "gebouw",
    ],
    "tr": [
        "bir", "bu", "ve", "için", "ile", "de", "da", "en", "çok",
        "olan", "gibi", "daha", "var", "ancak", "ama", "kadar", "sonra",
        "ya", "ben", "ne", "her", "olarak", "bunu", "olan", "büyük",
        "iyi", "gün", "yıl", "zaman", "şey", "kendi", "üzerinde",
        "önce", "oldu", "bütün", "nasıl", "yer", "göre", "başka",
        "arasında", "dünya", "ülke", "şehir", "hayat", "devlet",
        "hükümet", "toplum", "siyaset", "ekonomi", "eğitim",
        "Türkiye", "İstanbul", "Ankara", "güzel", "öğrenci",
        "üniversite", "çalışma", "bilgi", "kalite",
    ],
    "pl": [
        "i", "w", "na", "nie", "się", "z", "do", "to", "że", "jest",
        "jak", "ale", "o", "co", "po", "za", "od", "tak", "go", "przez",
        "by", "ten", "był", "jej", "już", "może", "pan", "tylko",
        "ze", "mnie", "tego", "tym", "mi", "jeszcze", "być", "tu",
        "sobie", "kiedy", "teraz", "bardzo", "też", "wszystko", "czas",
        "życie", "dzień", "świat", "kraj", "miasto", "praca", "dom",
        "państwo", "rząd", "społeczeństwo", "polityka", "edukacja",
        "informacja", "jakość", "Polska", "Warszawa", "Kraków",
        "łódź", "Gdańsk", "Wrocław", "źródło", "między",
    ],
}


def _discover_system_fonts() -> list[str]:
    """
    Return a list of TrueType/OpenType font paths available on the system.

    On macOS, looks in /System/Library/Fonts and /Library/Fonts.
    On Linux, looks in /usr/share/fonts.
    Only .ttf and .otf files are returned.
    """
    search_dirs: list[str] = []
    system = platform.system()

    if system == "Darwin":
        search_dirs = ["/System/Library/Fonts", "/Library/Fonts"]
    elif system == "Linux":
        search_dirs = ["/usr/share/fonts"]

    fonts: list[str] = []
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for dirpath, _, filenames in os.walk(search_dir):
            for fname in filenames:
                if fname.lower().endswith((".ttf", ".otf")):
                    fonts.append(os.path.join(dirpath, fname))

    return fonts


# Module-level cache so font discovery runs only once per process.
_SYSTEM_FONTS: Optional[list[str]] = None


def _get_system_fonts() -> list[str]:
    global _SYSTEM_FONTS
    if _SYSTEM_FONTS is None:
        _SYSTEM_FONTS = _discover_system_fonts()
    return _SYSTEM_FONTS


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font at the given size, falling back to the PIL default."""
    if font_path is not None:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


class SyntheticTextDataset(SizedDataset):
    """
    Procedurally generated text-line images for text recognition training.

    Each item is a dict matching the HierTextRecognition output format::

        {
            "image_id": str,      # e.g. "synth-00042"
            "image":    Tensor,   # [1, 64, W] float, values in [-0.5, 0.5]
            "text_seq": Tensor,   # [len(text)] int32, encoded class indices
            "text":     str,      # the raw text rendered in the image
        }

    Text is sampled from built-in word lists covering English, French, German,
    Spanish, Portuguese, Italian, Dutch, Turkish, and Polish, mixed with
    numbers, punctuation, and special characters. Images are rendered with
    random system fonts, sizes, and lightweight augmentations (rotation, blur,
    brightness jitter).
    """

    # Languages whose word lists are available.
    LANGUAGES = list(_WORD_LISTS.keys())

    def __init__(
        self,
        num_samples: int,
        alphabet: str,
        transform=None,
        max_images: Optional[int] = None,
        output_height: int = 64,
        seed: Optional[int] = None,
    ):
        """
        :param num_samples: Virtual size of the dataset.
        :param alphabet: String of characters the recognition model can
            output. Characters not in the alphabet are replaced by ``?``
            during encoding. The alphabet string is converted to a list
            internally for compatibility with ``encode_text``.
        :param transform: Optional torchvision transform applied to each
            image tensor *before* resizing to ``output_height``.
        :param max_images: If set, truncates the dataset to this many items.
        :param output_height: Height (in pixels) of every output image.
        :param seed: If set, each ``__getitem__`` call uses
            ``seed + idx`` as its RNG seed, making the dataset fully
            reproducible.
        """
        super().__init__()

        self._num_samples = min(num_samples, max_images) if max_images else num_samples
        self.alphabet: list[str] = list(alphabet)
        self.transform = transform
        self.output_height = output_height
        self._seed = seed

        # Pre-discover fonts once.
        self._fonts = _get_system_fonts()

        # Build a combined pool of words and fragments for sampling.
        self._word_pool: list[str] = []
        for lang_words in _WORD_LISTS.values():
            self._word_pool.extend(lang_words)
        self._word_pool.extend(_COMMON_NUMBERS)
        self._word_pool.extend(_SPECIAL_FRAGMENTS)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self._seed + idx if self._seed is not None else None)

        text = self._sample_text_line(rng)
        line_img = self._render_line(text, rng)
        img_tensor = self._pil_to_tensor(line_img)

        # Optional external augmentations (same interface as HierTextRecognition).
        if self.transform:
            img_tensor = self.transform(img_tensor)
            img_tensor = img_tensor.clamp(-0.5, 0.5)
            _, line_height, line_width = img_tensor.shape

        _, line_height, line_width = img_tensor.shape

        # Resize to fixed output height, preserving aspect ratio within limits.
        aspect_ratio = line_width / max(line_height, 1)
        output_width = min(800, max(10, int(self.output_height * aspect_ratio)))
        img_tensor = resize(
            img_tensor, [self.output_height, output_width], antialias=True
        )

        text_seq = encode_text(text, self.alphabet, unknown_char="?")

        return {
            "image_id": f"synth-{idx:06d}",
            "image": img_tensor,
            "text_seq": text_seq,
            "text": text,
        }

    # ------------------------------------------------------------------
    # Text sampling
    # ------------------------------------------------------------------

    def _sample_text_line(self, rng: random.Random) -> str:
        """
        Build a text line of 1-8 words/fragments.

        Approximately 15% of tokens are numbers or special fragments to
        ensure that the model sees diverse character classes.
        """
        n_words = rng.randint(1, 8)
        tokens: list[str] = []

        for _ in range(n_words):
            r = rng.random()
            if r < 0.10:
                # Insert a number token.
                tokens.append(rng.choice(_COMMON_NUMBERS))
            elif r < 0.15:
                # Insert a special fragment.
                tokens.append(rng.choice(_SPECIAL_FRAGMENTS))
            else:
                tokens.append(rng.choice(self._word_pool))

        line = " ".join(tokens)

        # Occasionally append trailing punctuation.
        if rng.random() < 0.3:
            line += rng.choice([".", ",", "!", "?", ":", ";", "..."])

        # Normalize so that the text is consistent with encode_text behavior.
        line = normalize_text(line)

        return line

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_line(self, text: str, rng: random.Random) -> Image.Image:
        """
        Render *text* onto a grayscale PIL image with random visual variation.

        Steps:
        1. Pick a random font and size.
        2. Draw text in a tight bounding box with small padding.
        3. Apply lightweight augmentations (rotation, blur, brightness).
        """

        font_size = rng.randint(16, 48)

        # Pick a random font.
        font_path: Optional[str] = None
        if self._fonts:
            font_path = rng.choice(self._fonts)
        font = _load_font(font_path, font_size)

        # Measure the text bounding box.
        # Use a temporary image to call textbbox.
        tmp = Image.new("L", (1, 1))
        tmp_draw = ImageDraw.Draw(tmp)
        bbox = tmp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Add padding around the text.
        pad_x = rng.randint(4, 16)
        pad_y = rng.randint(4, 12)
        img_width = max(text_width + 2 * pad_x, 10)
        img_height = max(text_height + 2 * pad_y, 10)

        # Random background and foreground brightness.
        bg_brightness = rng.randint(200, 255)
        fg_brightness = rng.randint(0, 60)
        if rng.random() < 0.15:
            # Invert: light text on dark background.
            bg_brightness, fg_brightness = fg_brightness, bg_brightness

        img = Image.new("L", (img_width, img_height), bg_brightness)
        draw = ImageDraw.Draw(img)

        # Draw the text. Offset by -bbox[0], -bbox[1] so the text starts at
        # the padding origin regardless of font metrics.
        draw.text(
            (pad_x - bbox[0], pad_y - bbox[1]),
            text,
            fill=fg_brightness,
            font=font,
        )

        # --- Augmentations ---

        # Slight rotation (-3 to 3 degrees).
        angle = rng.uniform(-3.0, 3.0)
        if abs(angle) > 0.3:
            img = img.rotate(
                angle,
                resample=Image.BILINEAR,
                expand=True,
                fillcolor=bg_brightness,
            )

        # Gaussian blur with random radius.
        if rng.random() < 0.3:
            radius = rng.uniform(0.3, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Brightness variation: scale pixel values around the mean.
        if rng.random() < 0.3:
            factor = rng.uniform(0.8, 1.2)
            arr = np.array(img, dtype=np.float32) * factor
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")

        return img

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """
        Convert a PIL grayscale image to a CHW float tensor in [-0.5, 0.5].

        Uses ``transform_image`` from ``.util`` for the normalisation step.
        """
        arr = np.array(img, dtype=np.uint8)
        # Shape: (H, W) -> (1, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return transform_image(tensor)

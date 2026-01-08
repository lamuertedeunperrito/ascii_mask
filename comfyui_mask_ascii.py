import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def to_u8(images: torch.Tensor) -> np.ndarray:
    imgs = images.detach().cpu().clamp(0, 1).numpy()
    return (imgs * 255).astype(np.uint8)


def from_u8(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images.astype(np.float32) / 255.0)


def to_mask(mask: torch.Tensor, hw):
    m = mask.detach().cpu().float()
    if m.ndim == 4 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim != 3:
        raise ValueError("MASK must be [B,H,W]")
    B, H, W = m.shape
    tH, tW = hw
    if (H, W) != (tH, tW):
        out = np.zeros((B, tH, tW), np.float32)
        for i in range(B):
            im = Image.fromarray((m[i].numpy() * 255).astype(np.uint8))
            im = im.resize((tW, tH), Image.NEAREST)
            out[i] = np.asarray(im) / 255.0
        return out
    return m.numpy()


class MaskASCIIStylizer:
    """
    ASCII Stylizer for IMAGE batches with MASK restriction
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "mask": ("MASK",),
                "cell_size": ("INT", {"default": 10, "min": 4, "max": 64}),
                "charset": ("STRING", {"default": " .:-=+*#%@"}),
                "contrast": ("FLOAT", {"default": 1.2, "min": 0.2, "max": 4.0}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0}),
                "invert": ("BOOLEAN", {"default": False}),
                "use_color": ("BOOLEAN", {"default": True}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "blend_mode": (["normal", "add", "screen"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "video/effects"

    def run(
        self,
        frames,
        mask,
        cell_size,
        charset,
        contrast,
        gamma,
        invert,
        use_color,
        opacity,
        blend_mode,
    ):
        frames_u8 = to_u8(frames)
        B, H, W, _ = frames_u8.shape
        mask_np = to_mask(mask, (H, W))

        chars = list(charset)
        n_chars = len(chars)

        out = np.zeros_like(frames_u8)

        # Simple monospaced font fallback
        font = ImageFont.load_default()

        for b in range(B):
            base = frames_u8[b]
            m = mask_np[b]

            canvas = Image.new("RGB", (W, H), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)

            for y in range(0, H, cell_size):
                for x in range(0, W, cell_size):
                    if m[y:y+cell_size, x:x+cell_size].mean() < 0.5:
                        continue

                    block = base[y:y+cell_size, x:x+cell_size]
                    if block.size == 0:
                        continue

                    # luminance
                    lum = (
                        0.2126 * block[..., 0]
                        + 0.7152 * block[..., 1]
                        + 0.0722 * block[..., 2]
                    ).mean() / 255.0

                    # tone mapping
                    lum = lum ** gamma
                    lum = np.clip((lum - 0.5) * contrast + 0.5, 0, 1)
                    if invert:
                        lum = 1.0 - lum

                    idx = int(lum * (n_chars - 1))
                    char = chars[idx]

                    if use_color:
                        col = tuple(np.mean(block.reshape(-1, 3), axis=0).astype(np.uint8))
                    else:
                        g = int(lum * 255)
                        col = (g, g, g)

                    draw.text((x, y), char, fill=col, font=font)

            ascii_img = np.asarray(canvas)

            if blend_mode == "normal":
                blended = (
                    base.astype(np.float32) * (1 - opacity)
                    + ascii_img.astype(np.float32) * opacity
                )
            elif blend_mode == "add":
                blended = np.clip(
                    base.astype(np.float32) + ascii_img.astype(np.float32) * opacity,
                    0,
                    255,
                )
            else:  # screen
                b0 = base.astype(np.float32) / 255.0
                a0 = ascii_img.astype(np.float32) / 255.0
                blended = (1 - (1 - b0) * (1 - a0 * opacity)) * 255.0

            out[b] = blended.astype(np.uint8)

        return (from_u8(out),)


NODE_CLASS_MAPPINGS = {
    "MaskASCIIStylizer": MaskASCIIStylizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskASCIIStylizer": "Mask ASCII Stylizer",
}

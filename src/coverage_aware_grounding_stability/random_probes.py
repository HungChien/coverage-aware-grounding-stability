from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class RandomProbe:
    family: str
    severity: float
    seed: int

    @property
    def name(self) -> str:
        return f"{self.family}_{self.severity:.6f}_seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)


def sample_random_probe(
    rng: np.random.Generator,
    distribution: Mapping[str, Mapping[str, float]],
) -> RandomProbe:
    """Sample one probe from a registered mixture distribution."""

    families = list(distribution)
    if not families:
        raise ValueError("the probe distribution is empty")
    weights = np.asarray(
        [float(distribution[name].get("weight", 1.0)) for name in families],
        dtype=float,
    )
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("probe-family weights must be non-negative and non-zero")
    weights /= weights.sum()
    family = str(rng.choice(families, p=weights))
    settings = distribution[family]
    low = float(settings["low"])
    high = float(settings["high"])
    if low > high:
        raise ValueError(f"invalid severity interval for {family}")
    severity = float(rng.uniform(low, high))
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    return RandomProbe(family=family, severity=severity, seed=seed)


def sample_stratified_probes(
    rng: np.random.Generator,
    distribution: Mapping[str, Mapping[str, float]],
    per_family: int,
) -> list[RandomProbe]:
    """Sample a balanced, round-robin probe registry.

    Every prefix containing a complete family block has equal family exposure.
    This makes family-risk attribution identifiable at small budgets and avoids
    accidental family imbalance being mistaken for model instability.
    """

    if per_family <= 0:
        raise ValueError("per_family must be positive")
    families = list(distribution)
    if not families:
        raise ValueError("the probe distribution is empty")

    by_family: dict[str, list[RandomProbe]] = {}
    for family in families:
        settings = distribution[family]
        low = float(settings["low"])
        high = float(settings["high"])
        if low > high:
            raise ValueError(f"invalid severity interval for {family}")
        by_family[family] = [
            RandomProbe(
                family=family,
                severity=float(rng.uniform(low, high)),
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            for _ in range(per_family)
        ]

    probes: list[RandomProbe] = []
    for repetition in range(per_family):
        family_order = list(families)
        rng.shuffle(family_order)
        probes.extend(by_family[family][repetition] for family in family_order)
    return probes


def apply_random_probe(image: Image.Image, probe: RandomProbe) -> Image.Image:
    """Apply a concrete random visual probe without changing image dimensions."""

    image = image.convert("RGB")
    if probe.family == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=probe.severity))
    if probe.family == "brightness":
        return ImageEnhance.Brightness(image).enhance(probe.severity)
    if probe.family == "jpeg":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=int(round(probe.severity)))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if probe.family == "resolution":
        width, height = image.size
        scale = probe.severity
        small = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        return small.resize((width, height), Image.Resampling.BILINEAR)
    if probe.family == "gaussian_noise":
        array = np.asarray(image, dtype=np.float32) / 255.0
        rng = np.random.default_rng(probe.seed)
        noise = rng.normal(0.0, probe.severity, size=array.shape)
        result = np.clip(array + noise, 0.0, 1.0)
        return Image.fromarray(np.uint8(np.round(result * 255.0)), mode="RGB")
    raise ValueError(f"unknown random probe family: {probe.family}")

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from PIL import Image

from .candidates import Candidate


class GroundingAdapter(ABC):
    @abstractmethod
    def predict(self, image: Image.Image, query: str, top_k: int) -> Sequence[Candidate]:
        raise NotImplementedError

    def predict_batch(
        self, images: Sequence[Image.Image], query: str, top_k: int
    ) -> list[Sequence[Candidate]]:
        """Default portable batch interface used by the benchmark runner."""

        return [self.predict(image, query, top_k) for image in images]


class GroundingDINOAdapter(GroundingAdapter):
    def __init__(
        self,
        model_name: str,
        box_threshold: float = 0.05,
        text_threshold: float = 0.05,
    ):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # The experiment uses the already cached, frozen checkpoint. Explicit
        # offline loading prevents a metadata request from changing run
        # availability while leaving model weights and inference unchanged.
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=True
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name, local_files_only=True
        )
        self.model.to(self.device).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, image: Image.Image, query: str, top_k: int) -> Sequence[Candidate]:
        text = query.strip().rstrip(".") + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = result.get("text_labels", result.get("labels", [""] * len(result["scores"])))
        candidates = [
            Candidate(tuple(map(float, box.tolist())), float(score), str(label))
            for box, score, label in zip(result["boxes"], result["scores"], labels)
        ]
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]

    def predict_batch(
        self, images: Sequence[Image.Image], query: str, top_k: int
    ) -> list[Sequence[Candidate]]:
        if not images:
            return []
        images = [image.convert("RGB") for image in images]
        text = query.strip().rstrip(".") + "."
        inputs = self.processor(
            images=images,
            text=[text] * len(images),
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1] for image in images],
        )
        batches: list[Sequence[Candidate]] = []
        for result in processed:
            labels = result.get(
                "text_labels", result.get("labels", [""] * len(result["scores"]))
            )
            candidates = [
                Candidate(tuple(map(float, box.tolist())), float(score), str(label))
                for box, score, label in zip(
                    result["boxes"], result["scores"], labels
                )
            ]
            batches.append(
                sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[
                    :top_k
                ]
            )
        return batches


def owlv2_result_to_candidates(
    result,
    query: str,
    top_k: int,
    image_size: tuple[int, int],
) -> list[Candidate]:
    """Convert one Transformers OWLv2 result to the benchmark contract.

    OWLv2 may emit coordinates a few pixels outside the unpadded image.  The
    adapter clips them to the observable image plane, drops degenerate boxes,
    and otherwise preserves the model's native confidence ordering.
    """

    width, height = image_size
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    labels = result.get("text_labels")
    if labels is None:
        labels = [query] * len(scores)
    candidates = []
    for box, score, label in zip(boxes, scores, labels):
        x0, y0, x1, y1 = map(float, box)
        clipped = (
            min(max(x0, 0.0), float(width)),
            min(max(y0, 0.0), float(height)),
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        candidates.append(Candidate(clipped, float(score), str(label or query)))
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[
        :top_k
    ]


def owlv2_text_context_length(model) -> int:
    """Return the text length supported by the loaded OWLv2 checkpoint."""

    text_config = getattr(model.config, "text_config", None)
    maximum = getattr(text_config, "max_position_embeddings", None)
    if not isinstance(maximum, int) or maximum <= 0:
        raise ValueError(
            "OWLv2 checkpoint does not expose a valid "
            "text_config.max_position_embeddings"
        )
    return maximum


class OWLv2Adapter(GroundingAdapter):
    """Hugging Face OWLv2 adapter using the same exposed-candidate contract."""

    def __init__(
        self,
        model_name: str,
        box_threshold: float = 0.05,
        revision: str | None = None,
    ):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        loading = {"local_files_only": True}
        if revision is not None:
            loading["revision"] = revision
        self.processor = Owlv2Processor.from_pretrained(model_name, **loading)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name, **loading)
        self.model.to(self.device).eval()
        self.text_context_length = owlv2_text_context_length(self.model)
        self.box_threshold = box_threshold
        self.revision = revision

    def predict(self, image: Image.Image, query: str, top_k: int) -> Sequence[Candidate]:
        return self.predict_batch([image], query, top_k)[0]

    def predict_batch(
        self, images: Sequence[Image.Image], query: str, top_k: int
    ) -> list[Sequence[Candidate]]:
        if not images:
            return []
        rgb_images = [image.convert("RGB") for image in images]
        text_labels = [[query.strip()] for _ in rgb_images]
        inputs = self.processor(
            images=rgb_images,
            text=text_labels,
            truncation=True,
            max_length=self.text_context_length,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor(
            [(image.height, image.width) for image in rgb_images]
        )
        processed = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.box_threshold,
            target_sizes=target_sizes,
            text_labels=text_labels,
        )
        return [
            owlv2_result_to_candidates(result, query, top_k, image.size)
            for result, image in zip(processed, rgb_images)
        ]


class YOLOWorldAdapter(GroundingAdapter):
    def __init__(self, model_name: str, box_threshold: float = 0.05):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.box_threshold = box_threshold
        self.current_query = None

    def predict(self, image: Image.Image, query: str, top_k: int) -> Sequence[Candidate]:
        if query != self.current_query:
            # Ultralytics caches CLIP as a submodule. After the detector is moved
            # to CUDA, the cached module parameters move but its `.device`
            # attribute can remain "cpu", causing newly tokenized prompts to be
            # placed on the wrong device. Synchronise that attribute before
            # encoding a new query.
            clip_model = getattr(self.model.model, "clip_model", None)
            if clip_model is not None:
                clip_model.device = next(clip_model.parameters()).device
            self.model.set_classes([query])
            self.current_query = query
        result = self.model.predict(
            source=image,
            conf=self.box_threshold,
            verbose=False,
            device=0,
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        candidates = [
            Candidate(tuple(map(float, box)), float(score), query)
            for box, score in zip(boxes, scores)
        ]
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]

    def predict_batch(
        self, images: Sequence[Image.Image], query: str, top_k: int
    ) -> list[Sequence[Candidate]]:
        if not images:
            return []
        if query != self.current_query:
            clip_model = getattr(self.model.model, "clip_model", None)
            if clip_model is not None:
                clip_model.device = next(clip_model.parameters()).device
            self.model.set_classes([query])
            self.current_query = query
        results = self.model.predict(
            source=[image.convert("RGB") for image in images],
            conf=self.box_threshold,
            verbose=False,
            device=0,
        )
        batches: list[Sequence[Candidate]] = []
        for result in results:
            if result.boxes is None:
                batches.append([])
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            candidates = [
                Candidate(tuple(map(float, box)), float(score), query)
                for box, score in zip(boxes, scores)
            ]
            batches.append(
                sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[
                    :top_k
                ]
            )
        return batches

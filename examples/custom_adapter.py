"""Minimal adapter for integrating a candidate-producing grounding model."""

from collections.abc import Sequence

from PIL import Image

from coverage_aware_grounding_stability import Candidate, GroundingAdapter


class MyGroundingAdapter(GroundingAdapter):
    """Replace ``backend`` calls with your model's native inference API."""

    def __init__(self, backend):
        self.backend = backend

    def predict(
        self,
        image: Image.Image,
        query: str,
        top_k: int,
    ) -> Sequence[Candidate]:
        predictions = self.backend.predict(image=image, text=query)
        candidates = [
            Candidate(
                box=tuple(float(value) for value in prediction["xyxy"]),
                score=float(prediction["score"]),
                label=str(prediction.get("label", query)),
            )
            for prediction in predictions
        ]
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]


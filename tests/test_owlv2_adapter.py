import pytest

torch = pytest.importorskip("torch")

from coverage_aware_grounding_stability.adapters import (
    owlv2_result_to_candidates,
    owlv2_text_context_length,
)


class Stub:
    pass


def test_owlv2_text_context_length_uses_loaded_checkpoint_contract():
    model = Stub()
    model.config = Stub()
    model.config.text_config = Stub()
    model.config.text_config.max_position_embeddings = 16

    assert owlv2_text_context_length(model) == 16


@pytest.mark.parametrize("maximum", [None, 0, -1, "16"])
def test_owlv2_text_context_length_rejects_invalid_contract(maximum):
    model = Stub()
    model.config = Stub()
    model.config.text_config = Stub()
    model.config.text_config.max_position_embeddings = maximum

    with pytest.raises(ValueError, match="max_position_embeddings"):
        owlv2_text_context_length(model)


def test_owlv2_candidates_are_clipped_filtered_sorted_and_truncated():
    result = {
        "boxes": torch.tensor(
            [
                [-2.0, -3.0, 12.0, 13.0],
                [20.0, 20.0, 15.0, 25.0],
                [1.0, 1.0, 5.0, 5.0],
            ]
        ),
        "scores": torch.tensor([0.4, 0.9, 0.8]),
        "text_labels": ["target", "target", "target"],
    }

    candidates = owlv2_result_to_candidates(
        result, query="target", top_k=2, image_size=(10, 10)
    )

    assert len(candidates) == 2
    assert candidates[0].box == (1.0, 1.0, 5.0, 5.0)
    assert candidates[1].box == (0.0, 0.0, 10.0, 10.0)
    assert candidates[0].score > candidates[1].score
    assert all(candidate.label == "target" for candidate in candidates)


def test_owlv2_candidates_fall_back_to_query_label():
    result = {
        "boxes": torch.tensor([[0.0, 0.0, 3.0, 4.0]]),
        "scores": torch.tensor([0.5]),
    }

    candidates = owlv2_result_to_candidates(
        result, query="red cup", top_k=20, image_size=(10, 10)
    )

    assert candidates[0].label == "red cup"

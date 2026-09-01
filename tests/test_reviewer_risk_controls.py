import pytest

from scripts.analyse_reviewer_risk_controls import normalised_strength


@pytest.mark.parametrize(
    "family,mild,severe",
    [
        ("blur", 0.3, 2.5),
        ("jpeg", 95.0, 40.0),
        ("resolution", 1.0, 0.5),
        ("gaussian_noise", 0.0, 0.04),
    ],
)
def test_normalised_strength_endpoints(family, mild, severe):
    assert normalised_strength(family, mild) == pytest.approx(0.0)
    assert normalised_strength(family, severe) == pytest.approx(1.0)


def test_brightness_strength_is_two_sided():
    assert normalised_strength("brightness", 1.0) == pytest.approx(0.0)
    assert normalised_strength("brightness", 0.7) == pytest.approx(1.0)
    assert normalised_strength("brightness", 1.3) == pytest.approx(1.0)

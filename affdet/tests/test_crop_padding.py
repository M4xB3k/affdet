import pytest
from affdet.extract.dinov2 import _pad_bbox, DINOv2Extractor


def test_zero_padding_is_identity():
    assert _pad_bbox([10, 20, 30, 60], 100, 100, 0.0) == [10, 20, 30, 60]


def test_pad_expands_by_fraction():
    # w=20,h=40; 10% -> +2 x each side, +4 y each side
    assert _pad_bbox([10, 20, 30, 60], 100, 100, 0.1) == [8, 16, 32, 64]


def test_pad_clamps_to_image_bounds():
    # near edges -> clamp to 0 and W/H
    assert _pad_bbox([2, 2, 98, 98], 100, 100, 0.5) == [0, 0, 100, 100]


def test_extractor_defaults_zero_padding():
    assert DINOv2Extractor().crop_padding == 0.0

import numpy as np
from affdet.render.mask_export import mask_to_bbox, extract_bboxes_from_pass


def test_mask_to_bbox_simple():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 2:6] = 1
    x1, y1, x2, y2 = mask_to_bbox(mask)
    assert x1 == 2 and y1 == 3 and x2 == 5 and y2 == 6


def test_extract_bboxes_from_multi_object_pass():
    pass_img = np.zeros((50, 50), dtype=np.uint8)
    pass_img[5:15, 5:15] = 1
    pass_img[20:35, 10:30] = 2
    pass_img[40:48, 40:48] = 3
    bboxes = extract_bboxes_from_pass(pass_img)
    assert set(bboxes.keys()) == {1, 2, 3}
    assert bboxes[1] == (5, 5, 14, 14)


def test_bbox_padding():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1
    x1, y1, x2, y2 = mask_to_bbox(mask, padding=0.1, img_size=(100, 100))
    assert x1 < 40 and y1 < 40 and x2 > 59 and y2 > 59

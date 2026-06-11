"""YOLO dataset formatting: PNG scaling/flip and label normalization (offline)."""

import numpy as np
import pandas as pd
import pytest

from solarflare.detect.dataset import magnetogram_to_png_array, yolo_label_lines


class TestPngConversion:
    def test_scaling_and_nan(self):
        data = np.array([[-600.0, 0.0], [300.0, np.nan]], dtype=np.float32)
        png = magnetogram_to_png_array(data, clip_gauss=300.0)
        assert png.dtype == np.uint8
        # flipud: original row 0 is now row 1
        assert png[1, 0] == 0          # -600 clipped to -300 -> 0
        assert png[1, 1] == 127        # 0 G -> mid-gray
        assert png[0, 0] == 255        # +300 -> 255
        assert png[0, 1] == 127        # NaN -> neutral

    def test_flip_puts_fits_bottom_at_png_bottom_row(self):
        data = np.zeros((4, 3), dtype=np.float32)
        data[0, :] = 300.0             # FITS bottom row bright
        png = magnetogram_to_png_array(data)
        assert (png[3] == 255).all()   # ends up in last PNG row (image bottom)


class TestYoloLabels:
    def test_normalization_and_flip(self):
        boxes = pd.DataFrame([{"x_min": 10.0, "x_max": 30.0, "y_min": 70.0, "y_max": 90.0}])
        (line,) = yolo_label_lines(boxes, width=100, height=100, min_box_px=2)
        cls, cx, cy, w, h = line.split()
        assert cls == "0"
        assert float(cx) == pytest.approx(0.205, abs=1e-3)   # (20 + 0.5)/100
        # FITS cy=80 -> PNG cy = 1 - 0.805
        assert float(cy) == pytest.approx(0.195, abs=1e-3)
        assert float(w) == pytest.approx(0.2, abs=1e-3)
        assert float(h) == pytest.approx(0.2, abs=1e-3)

    def test_clipping_to_frame(self):
        boxes = pd.DataFrame([{"x_min": -20.0, "x_max": 30.0, "y_min": 50.0, "y_max": 140.0}])
        (line,) = yolo_label_lines(boxes, width=100, height=100, min_box_px=2)
        _, cx, cy, w, h = map(float, line.split())
        assert 0.0 <= cx - w / 2 and cx + w / 2 <= 1.0
        assert 0.0 <= cy - h / 2 and cy + h / 2 <= 1.0

    def test_min_size_filter(self):
        boxes = pd.DataFrame([
            {"x_min": 0.0, "x_max": 3.0, "y_min": 0.0, "y_max": 50.0},   # too thin
            {"x_min": 10.0, "x_max": 40.0, "y_min": 10.0, "y_max": 40.0},
        ])
        lines = yolo_label_lines(boxes, width=100, height=100, min_box_px=5)
        assert len(lines) == 1

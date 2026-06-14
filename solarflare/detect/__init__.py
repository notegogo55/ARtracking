"""Stage B1: AR detection & segmentation.

A pluggable `Segmenter` registry (`segmenter`) resolves `segment.model`
(threshold | unet | surya | sam2) to a class; YOLO provides the no-SHARP
detection path. Implemented in Phase B.
"""

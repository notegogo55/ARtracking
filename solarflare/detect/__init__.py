"""Stage B: AR segmentation & tracking (the magnetic-root mask).

A pluggable `Segmenter` registry (`segmenter`) resolves `segment.model`
(threshold | unet | surya | sam2) to a class that segments each AR ONCE on the
HMI magnetogram. `bootstrap` derives HARP boxes (no hand-labeling); `fulldisk`
fetches the bounded full-disk frames the operational views overlay. Implemented
in Phase B.
"""

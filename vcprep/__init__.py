"""Speech dataset preprocessing for voice conversion training.

    HF datasets -> UVR (Kim Vocal 2) -> silence removal -> quality scoring
                -> clean / low_quality

Fine-tuning lives in the sibling :mod:`seedvc_ft` package; the two are fully
decoupled and share only the on-disk output layout.
"""
__version__ = "0.1.0"

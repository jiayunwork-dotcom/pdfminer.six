"""Minimal offline stand-in for charset_normalizer used only in this sandbox.

The real third-party package is unavailable here.  pdfminer only needs
``detect`` for best-effort bytes decoding; utf-8 with replacement is enough
for the test suite and the demos.
"""


def detect(byte_str):
    return {"encoding": "utf-8", "language": "", "confidence": 0.0}

"""
Entry point alias for ``python -m resonantbrain``.

Delegates to :func:`resonantbrain.main.main`.
"""

from .main import main

if __name__ == "__main__":
    main()

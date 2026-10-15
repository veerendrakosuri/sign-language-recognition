"""Support ``python -m signlang`` as well as the ``signlang`` entry point."""
from .cli import main

raise SystemExit(main())

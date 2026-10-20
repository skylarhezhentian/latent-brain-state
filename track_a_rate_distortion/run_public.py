"""Run the Track A pilot against the public IBL OpenAlyx database.

The analysis module contains the scientific protocol. This entry point adds the
public OpenAlyx username required by recent ONE-api releases, while leaving the
analysis implementation reusable in an already configured local environment.
"""
from __future__ import annotations

from one.api import ONE as RealONE

import track_a_rate_distortion.run_track_a as analysis


def public_one(*args, **kwargs):
    """Instantiate ONE using the documented public OpenAlyx account name."""
    kwargs.setdefault("username", "intbrainlab")
    return RealONE(*args, **kwargs)


analysis.ONE = public_one

if __name__ == "__main__":
    analysis.main()

"""Test configuration shared by the tests of all packages of the monorepo.

The `ci` profile of Hypothesis, loaded with `HYPOTHESIS_PROFILE=ci` by the GitHub
Actions workflows, has no deadline per example: the first examples of a run in a fresh
environment, e.g. importing and compiling large dependencies, can exceed any deadline.
It prints how to reproduce a failing example, as runs on CI start without the example
database of earlier runs.
"""

import os

try:
    from hypothesis import settings
except ImportError:  # e.g. a package whose tests do not use Hypothesis
    pass
else:
    settings.register_profile("ci", deadline=None, print_blob=True)
    settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))

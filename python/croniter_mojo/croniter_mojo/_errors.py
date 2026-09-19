"""croniter_mojo exception hierarchy.

Names mirror the public exception classes of the `croniter` PyPI package
(the black-box oracle for this project) so that calling code can catch the
same error categories; they are independent classes, not subclasses of the
oracle's. All inherit from :class:`CroniterMojoError`.
"""

from __future__ import annotations


class CroniterMojoError(Exception):
    """Base class for all croniter_mojo errors."""


class CroniterBadCronError(CroniterMojoError):
    """The cron expression is malformed or has out-of-range values."""


class CroniterNotAlphaError(CroniterBadCronError):
    """A field contains a token that is neither numeric nor a known name."""


class CroniterUnsupportedSyntaxError(CroniterBadCronError):
    """The expression uses syntax outside the supported 5-field scope."""


class CroniterBadDateError(CroniterMojoError):
    """No matching datetime exists within the search horizon."""

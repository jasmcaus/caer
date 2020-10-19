#pylint:disable=redefined-outer-name,attribute-defined-outside-init
"""
Timezone-related classes and functions.
"""

from datetime import datetime, timedelta, timezone


def get_fixed_timezone(offset):
    """Return a tzinfo instance with a fixed offset from UTC."""
    if isinstance(offset, timedelta):
        offset = offset.total_seconds() // 60
    sign = '-' if offset < 0 else '+'
    hhmm = '%02d%02d' % divmod(abs(offset), 60)
    name = sign + hhmm
    return timezone(timedelta(minutes=offset), name)


def _get_timezone_name(timezone):
    """Return the name of ``timezone``."""
    return timezone.tzname(None)


# Utilities

def now():
    """
    Return a naive datetime.datetime.
    """
    return datetime.now()
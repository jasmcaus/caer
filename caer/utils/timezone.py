#pylint:disable=redefined-outer-name,attribute-defined-outside-init
"""
Timezone-related classes and functions.
"""

from contextlib import ContextDecorator
from datetime import datetime, timedelta, timezone, tzinfo

import pytz
from asgiref.local import Local


# UTC time zone as a tzinfo instance.
utc = pytz.utc


def get_fixed_timezone(offset):
    """Return a tzinfo instance with a fixed offset from UTC."""
    if isinstance(offset, timedelta):
        offset = offset.total_seconds() // 60
    sign = '-' if offset < 0 else '+'
    hhmm = '%02d%02d' % divmod(abs(offset), 60)
    name = sign + hhmm
    return timezone(timedelta(minutes=offset), name)


_active = Local()


def _get_timezone_name(timezone):
    """Return the name of ``timezone``."""
    return timezone.tzname(None)


# Timezone selection functions.

# These functions don't change os.environ['TZ'] and call time.tzset()
# because it isn't thread safe.


def activate(timezone):
    """
    Set the time zone for the current thread.
    The ``timezone`` argument must be an instance of a tzinfo subclass or a
    time zone name.
    """
    if isinstance(timezone, tzinfo):
        _active.value = timezone
    elif isinstance(timezone, str):
        _active.value = pytz.timezone(timezone)
    else:
        raise ValueError("Invalid timezone: %r" % timezone)


def deactivate():
    """
    Unset the time zone for the current thread..
    """
    if hasattr(_active, "value"):
        del _active.value


class override(ContextDecorator):
    """
    Temporarily set the time zone for the current thread.
    This is a context manager that uses django.utils.timezone.activate()
    to set the timezone on entry and restores the previously active timezone
    on exit.
    The ``timezone`` argument must be an instance of a ``tzinfo`` subclass, a
    time zone name, or ``None``. If it is ``None``, Django enables the default
    time zone.
    """
    def __init__(self, timezone):
        self.timezone = timezone

    def __enter__(self):
        self.old_timezone = getattr(_active, 'value', None)
        if self.timezone is None:
            deactivate()
        else:
            activate(self.timezone)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_timezone is None:
            deactivate()
        else:
            _active.value = self.old_timezone


# Utilities

def now():
    """
    Return a naive datetime.datetime.
    """
    return datetime.now()
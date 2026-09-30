"""Shared date parsing and calendar-period helpers backed by python-dateutil.

The month helpers all take and return whole ``date`` objects and go through
``relativedelta``, so month-length and leap-year edges are handled in one place
instead of being re-derived (as ``year * 12 + month`` arithmetic, or as "first
of next month minus a day") wherever a month window is needed.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

from dateutil import parser
from dateutil.relativedelta import relativedelta

IST = ZoneInfo("Asia/Kolkata")

# The trailing window every month-over-month chart on the site defaults to. One
# name so the service, the route and the widget cannot drift to different
# windows and disagree about how much history is on screen.
DEFAULT_TREND_MONTHS = 12


def month_start(day: date, back: int = 0) -> date:
    """The first of the month ``back`` months before ``day``'s month.

    ``back=0`` is ``day``'s own month. ``back`` may be negative to move forward.
    """
    return (day - relativedelta(months=back)).replace(day=1)


def month_end(day: date) -> date:
    """The last day of ``day``'s month, whatever its length."""
    return day.replace(day=1) + relativedelta(months=1, days=-1)


def quarter_start(day: date) -> date:
    """The first day of the calendar quarter that contains ``day``."""
    first_month = (day.month - 1) // 3 * 3 + 1
    return day.replace(month=first_month, day=1)


def financial_year_start(day: date) -> date:
    """The first day of the Indian financial year that contains ``day``.

    The financial year runs April to March. A day in January to March
    belongs to the year that started the previous April.
    """
    year = day.year if day.month >= 4 else day.year - 1
    return date(year, 4, 1)


def month_key(day: date) -> str:
    """``day``'s month as ``YYYY-MM`` — the key format the trend series use."""
    return f"{day.year:04d}-{day.month:02d}"


def trailing_month_starts(day: date, months: int) -> list[date]:
    """The firsts of the trailing ``months`` calendar months, oldest first.

    The window *ends* with ``day``'s own month, which is included even though it
    is normally partial, so the newest point of a trend is the month in progress.
    """
    return [month_start(day, back) for back in range(months - 1, -1, -1)]


def parse_datetime(
    value: str | datetime | None, *, dayfirst: bool = False
) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    return parser.parse(text, dayfirst=dayfirst)


def parse_date(
    value: str | date | datetime | None, *, dayfirst: bool = False
) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = parse_datetime(value, dayfirst=dayfirst)
    return parsed.date() if parsed else None


def format_ddmmyyyy(value: date | datetime | None) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        value = value.date()
    return value.strftime("%d/%m/%Y")


def email_received_at_ist(received_at: datetime) -> datetime:
    """Convert an email Date header to IST.

    An email that arrives immediately after midnight IST has a different IST
    date than its UTC date. A header with an unknown zone gives no time zone.
    Such a header is IST.

    Args:
        received_at: The parsed Date header, aware or naive.

    Returns:
        The same moment as an aware IST datetime.
    """
    if received_at.tzinfo is None:
        return received_at.replace(tzinfo=IST)
    return received_at.astimezone(IST)

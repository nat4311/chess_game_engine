import datetime
from zoneinfo import ZoneInfo

MONTHS_LONG = [
    None,
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

MONTHS_SHORT = [
    None,
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
]

def pretty_datetime(style = 0, timezone = "America/Chicago") -> str:
    """
    check zoneinfo.ZoneInfo for info on timezone input
    ---------------------
    STYLE    EXAMPLE
    0        2025-8-14__8-34-19
    1        Aug-14-2025__8-34-19
    2        August-14-2025__8-34-19
    """
    dt = datetime.datetime.now(ZoneInfo(timezone))
    h = str(dt.hour).rjust(2,'0')
    m = str(dt.minute).rjust(2,'0')
    s = str(dt.second).rjust(2,'0')

    if style == 0:
        dt_str = f"{dt.year}-{dt.month}-{dt.day}__{h}-{m}-{s}"
    elif style == 1:
        dt_str = f"{MONTHS_SHORT[dt.month]}-{dt.month}-{dt.year}__{h}-{m}-{s}"
    elif style == 2:
        dt_str = f"{MONTHS_LONG[dt.month]}-{dt.month}-{dt.year}__{h}-{m}-{s}"
    else:
        raise ValueError(f"invalid style: {style} (type {type(style)})")
    return dt_str

def pretty_time_elapsed(t0: float, t1: float, style = 0) -> str:
    """
    t0 and t1 are start and end time in seconds
    rounds to nearest second
    --------------------------------------------
    STYLE    EXAMPLE
    0        03:31:26
    1        03H:31M:26S
    2        3 hr, 31 min, 26 s
    3        3 hours, 31 minutes, 26 seconds
    """
    t = round(t1 - t0)
    days = int(t/86400)
    t -= days * 86400
    hours = int(t/3600)
    t -= hours * 3600
    minutes = int(t/60)
    t -= minutes * 60
    seconds = t

    if style == 0:
        s = str(seconds).rjust(2,'0')
        m = str(minutes).rjust(2,'0')
        h = str(hours).rjust(2,'0')
        time_string = f"{h}:{m}:{s}"
        if days > 0:
            time_string = f"{days}:" + time_string
    elif style == 1:
        s = str(seconds).rjust(2,'0')
        m = str(minutes).rjust(2,'0')
        h = str(hours).rjust(2,'0')
        time_string = f"{h}H:{m}M:{s}S"
        if days > 0:
            time_string = f"{days}D:" + time_string
    elif style == 2:
        time_string = f"{hours} hr, {minutes} min, {seconds} s"
        if days == 1:
            time_string = "1 day, " + time_string
        elif days > 1:
            time_string = f"{days} days, " + time_string
    elif style == 3:
        s = f"{seconds} second{'' if seconds==1 else 's'}"
        m = f"{minutes} minute{'' if minutes==1 else 's'}"
        h = f"{hours} hour{'' if hours==1 else 's'}"
        d = f"{days} day{'' if days==1 else 's'}"
        if days > 0:
            time_string = f"{d}, {h}, {m}, {s}"
        else:
            time_string = f"{h}, {m}, {s}"
    else:
        raise ValueError(f"invalid style: {style} (type {type(style)})")

    return time_string

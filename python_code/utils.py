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

import datetime
import pandas as pd


def is_empty(x):

    if x is None:
        return True
    
    if isinstance(x, pd.DataFrame):
        return x.empty
     
    return not bool(x)


def parse_date(dt, fmt = "%Y-%m-%d"):
        
    if dt is None:
        raise TypeError('Cannot parse None to date')

    if isinstance(dt, str):
        try:
            parsed = pd.to_datetime(dt, format = fmt)
        except Exception as e:
            msg = "Cannot parse string {} to timestamp. Error: {}"
            msg = msg.format(dt, str(e))
            raise type(e)(msg)
        
        return parsed

    if isinstance(dt, datetime.datetime) or isinstance(dt, datetime.date):
        return pd.to_datetime(dt)
    
    if isinstance(dt, pd.Timestamp):
        return dt

    raise TypeError('Cannot parse type {} to date'.format(type(dt).__name__))


def quote(s: str):
    return "'" + s + "'"


def clamp(x, low, high):
    if x > high:
        x = high
    elif x < low:
        x = low

    return x

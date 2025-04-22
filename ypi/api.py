import uvicorn
import pandas as pd

from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from more_itertools import partition
from typing import List, Optional

from .yahoo import (
    parse_date,
    get_symbol_info_group,
    get_symbol_extra_data,
    get_price_group,
    get_financials,
    Success,
    Error,
)

from .util import clamp


class Response(BaseModel):
    success: Success
    error: List[str]
    
    class Config:
        arbitrary_types_allowed = True


router = APIRouter()
MAX_TIMEOUT = 600


@router.get("/symbols", response_model = Response)
async def symbols(symbol: str, timeout: int = 60, chunk: int = 20) -> Response:

    timeout = clamp(timeout, 0, MAX_TIMEOUT)
    symbols = parse_symbol(symbol)
    if not symbols:
        return Response(success = Success([]), error = [])
    
    if chunk > 50:
        e = Error(chunk, ValueError(f"Parameter chunk is too big: {chunk}"))
        raise HTTPException(status_code = 500, detail = repr(e))

    try:
        symbols = set(symbols)
        output = await get_symbol_info_group(symbols, timeout = timeout, chunk = chunk)
    except Exception as err:
        err = Error("there was an error when getting the symbol data", err)
        params = dict(symbol = symbols, timeout = timeout, chunk = chunk)
        detail = dict(params = params, error = repr(err))
        raise HTTPException(status_code = 500, detail = detail)

    e, s = partition(lambda x: isinstance(x, Success), output)
    e = [repr(e) for e in e]
    s = [s.data for s in s]
    s.sort(key = lambda x: x.ticker)
    s = Success(s)
    return Response(success = s, error = e)


@router.get("/symbol_extra", response_model = Response)
async def symbol_extra(symbol: str, timeout: int = 60, chunk: int = 20) -> Response:

    timeout = clamp(timeout, 0, MAX_TIMEOUT)
    symbols = parse_symbol(symbol)
    if not symbols:
        return Response(success = Success([]), error = [])
    
    if chunk > 50:
        e = Error(chunk, ValueError(f"Parameter chunk is too big: {chunk}"))
        raise HTTPException(status_code = 500, detail = repr(e))

    try:
        symbols = set(symbols)
        output = await get_symbol_extra_data(symbols, timeout = timeout, chunk = chunk)
    except Exception as err:
        err = Error("there was an error when getting the symbol extra data", err)
        params = dict(symbol = symbols, timeout = timeout, chunk = chunk)
        detail = dict(params = params, error = repr(err))
        raise HTTPException(status_code = 500, detail = detail)

    e, s = partition(lambda x: isinstance(x, Success), output)
    e = [repr(e) for e in e]
    s = [s.data for s in s]
    s.sort(key = lambda x: x.ticker)
    s = Success(s)
    return Response(success = s, error = e)


@router.get("/prices", response_model = Response)
async def prices(
    symbol: str,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
    interval: str = "1d",
    actions: bool = True,
    timeout: int = 60,
    chunk: int = 20
) -> Response:

    timeout = clamp(timeout, 0, MAX_TIMEOUT)
    symbols = parse_symbol(symbol)
    if not symbols:
        return Response(success = Success({}), error = [])
    
    symbols = set(symbols)
    end = pd.Timestamp.today() if (endDate is None) else parse_date(endDate)
    bgn = (end - pd.Timedelta(days = 5)) if (startDate is None) else parse_date(startDate)

    if bgn > end:
        e = Error(repr((bgn, end)), ValueError("Start date cannot be after end date"))
        raise HTTPException(status_code = 500, detail = repr(e))
    
    if chunk > 50:
        e = Error(chunk, ValueError(f"Parameter chunk is too big: {chunk}"))
        raise HTTPException(status_code = 500, detail = repr(e))

    try:
        output = await get_price_group(
            symbols,
            start_date = bgn,
            end_date = end,
            interval = interval,
            actions = actions,
            timeout = timeout,
            chunk = chunk
        )
    except Exception as err:
        params = dict(
            symbol = symbols,
            start_date = bgn,
            end_date = end,
            interval = interval,
            actions = actions,
            timeout = timeout,
            chunk = chunk,
        )
        err = Error("there was an error when grabbing the prices", err)
        detail = dict(params = params, error = repr(err))
        raise HTTPException(status_code = 500, detail = detail)

    e, s = partition(lambda x: isinstance(x, Success), output)
    e = [repr(e) for e in e]

    maps = {}
    for dct in s:
        for key, val in dct.data.items():
            maps[key] = val
            maps[key].sort(key = lambda x: x.date)
    
    s = Success(maps)
    return Response(success = s, error = e)


@router.get("/financials", response_model = Response)
async def financials(symbol: str, freq: str = "yearly", timeout: int = 60) -> Response:

    timeout = clamp(timeout, 0, MAX_TIMEOUT)
    sym = "" if (symbol is None) else symbol
    if sym == "":
        msg = "Symbol must not be empty nor None"
        e = Error(symbol, ValueError(msg))
        raise HTTPException(status_code = 500, detail = repr(e))

    f = freq.lower()[0]
    if f not in ("y", "q"):
        msg = "Frequency must be either 'yearly' or 'quarterly'"
        e = Error(freq, ValueError(msg))
        raise HTTPException(status_code = 500, detail = repr(e))

    freq = dict(y = "yearly", q = "quarterly").get(f, "unknown")
    try:
        output = await get_financials(sym, timeout = timeout, freq = freq)
    except Exception as err:
        err = Error(f"Runtime error when getting the financials for symbol: {symbol}", err)
        err = repr(err)
        return Response(success = Success({}), error = [err])

    errors = []
    success = dict()
    while output:
        key, val = output.popitem()
        if isinstance(val, Error):
            msg = f"financials: {key}, symbol: {symbol}"
            err = Error(msg, val.error)
            errors.append(repr(err))
            continue

        success[key] = val.data

    return Response(success = Success(success), error = errors)


def create_app(root_path = None):
    root_path = "/" if root_path == "" else root_path
    if root_path[0] != "/":
        root_path = "/" + root_path

    print(root_path)
    app = FastAPI(root_path = root_path)
    app.include_router(router)
    return app


def serve(host = "127.0.0.1", port = 8080, **kws):
    app = create_app(root_path = kws.get("root_path"))
    uvicorn.run(app, host = host, port = port, factory = False)


def parse_symbol(symbol):
    return [str(s).strip().upper() for s in symbol.split(",") if s != ""]

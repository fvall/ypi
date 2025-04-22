import asyncio
import logging
import datetime
import itertools as it
import yfinance as yf
import pandas as pd

from typing import Optional, Any
from pydantic import BaseModel
from more_itertools import always_iterable, chunked, partition
from .util import parse_date, is_empty


logger = logging.getLogger(__name__)


class ResponseSymbol(BaseModel):
    ticker: str
    symbol: str
    name: Optional[str]
    currency: Optional[str]
    quote_type: Optional[str]
    sector: Optional[str]
    industry: Optional[str]


class ResponseSymbolExtra(BaseModel):
    ticker: str
    symbol: str
    beta: float
    dividend_yield: float
    float_shares: float
    forward_pe: float
    market_cap: float
    shares_outstanding: float
    trailing_eps: float
    trailing_pe: float


class ResponsePrice(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    dividend: float
    split: float


class Success(BaseModel):
    data: Any

    def __init__(self, data):
        super().__init__(data = data)
    
    def __repr__(self):
        return f"Success(data={repr(self.data)})"


class Error(BaseModel):
    data: Any
    error: Exception

    def __init__(self, data, error):
        super().__init__(data = data, error = error)

    def __repr__(self):
        return f"Error(data={repr(self.data)}, error={repr(self.error)})"

    class Config:
        arbitrary_types_allowed = True


def is_short(x, length):
    return len(x) < length


def _get_info(ticker):
    logger.info(f"Getting ticker information for {ticker.ticker}")
    data = ticker.get_info()
    logger.info(f"Getting ticker information for {ticker.ticker} - complete")
    return data


async def get_symbol_info(symbol, timeout = 60):
    ticker = yf.Ticker(symbol.upper())
    task = asyncio.wait_for(asyncio.to_thread(_get_info, ticker), timeout)
    task = asyncio.create_task(task, name = ticker.ticker)
    return task


def _get_hist_price(
    ticker,
    bgn,
    end,
    interval = "1d",
    actions = True,
):
    logger.info(f"Getting price for ticker: {ticker.ticker}")
    prices = ticker.history(start = bgn, end = end, auto_adjust = False, actions = actions, interval = interval)
    logger.info(f"Getting price for ticker: {ticker.ticker} - complete")
    return prices


async def get_price(symbol, bgn, end, interval = "1d", actions = True, timeout = 60):
    ticker = yf.Ticker(symbol.upper())
    task = asyncio.wait_for(
        asyncio.to_thread(
            _get_hist_price,
            ticker,
            bgn,
            end,
            interval = interval,
            actions = actions,
        ),
        timeout
    )

    task = asyncio.create_task(task, name = ticker.ticker)
    return task


def _format_information_result(info, ticker):
    if isinstance(info, Exception):
        return Error(ticker, info)
        
    length = 2
    if is_short(info, length):
        return Error(ticker, ValueError(f"Symbol information has length less than {length}"))

    output = ResponseSymbol(
        ticker = ticker,
        symbol = info['symbol'],
        name = info.get('shortName'),
        currency = info.get("currency"),
        quote_type = info.get("quoteType"),
        sector = info.get("sector"),
        industry = info.get("industry"),
    )
    return Success(output)


def _format_information_extra_result(info, ticker):
    if isinstance(info, Exception):
        return Error(ticker, info)
        
    length = 2
    if is_short(info, length):
        return Error(ticker, ValueError(f"Symbol information has length less than {length}"))

    output = ResponseSymbolExtra(
        ticker = ticker,
        symbol = info['symbol'],
        shares_outstanding = info.get('sharesOutstanding', 0),
        float_shares = info.get('floatShares', 0),
        beta = info.get("beta", 0.0),
        dividend_yield = info.get("dividendYield", 0.0),
        forward_eps = info.get("forwardEps", 0.0),
        trailing_eps = info.get("trailingEps", 0.0),
        forward_pe = info.get("forwardPE", 0.0),
        trailing_pe = info.get("trailingPE", 0.0),
        market_cap = info.get("marketCap", 0.0),
    )
    return Success(output)


def _format_price_result(prices, ticker):
    if isinstance(prices, Exception):
        return Error(ticker, prices)
        
    if is_empty(prices):
        return Success([])

    want = [
        "Open",
        "High",
        "Close",
        "Low",
        "Volume",
        "Adj Close",
        "Dividends",
        "Stock Splits",
    ]

    bad = []
    for col in want:
        if col not in prices.columns:
            bad.append(col)

    if bad:
        cols = ", ".join(bad)
        return Error(prices, ValueError(f"Prices for ticker {ticker} are missing the following columns: {cols}"))

    if not isinstance(prices.index, pd.DatetimeIndex):
        klass = prices.index.__class__.__name__
        return Error(klass, TypeError(f"Index for prices should be DatetimeIndex but are: {klass}"))

    output = [None] * len(prices)
    for i, (idx, row) in enumerate(prices.iterrows()):
        try:
            date = idx.isoformat()
        except Exception as err:
            msg = f"Could not format date: {repr(date)} into isoformat for ticker {ticker}"
            logger.error(msg)
            logger.error(f"Error: {repr(err)}")
            return Error(prices, RuntimeError(msg))

        resp = ResponsePrice(
            date = date,
            open = row.get("Open", 0),
            high = row.get("High", 0),
            low  = row.get("Low", 0),
            close = row.get("Close", 0),
            adj_close = row.get("Adj Close", 0),
            volume = row.get("Volume", 0),
            dividend = row.get("Dividends", 0),
            split = row.get("Stock Splits", 0),
        )

        output[i] = resp

    return Success(output)


async def get_symbol_info_group(symbols, timeout = 60, chunk = 20):

    symbols = set(always_iterable(symbols))
    output = []
    if not symbols:
        data = Error(symbols, ValueError("Parameter symbols cannot be empty"))
        output.append(data)

    if chunk > 50:
        data = Error(chunk, ValueError(f"Parameter chunk is too big: {chunk}"))
        output.append(data)

    if output:
        return output

    full_timeout = chunk * timeout
    if full_timeout <= 0:
        full_timeout = None

    if timeout <= 0:
        timeout = None

    for sym in chunked(sorted(symbols), chunk):
        tasks = [get_symbol_info(s, timeout) for s in sym]
        for res in asyncio.as_completed(tasks, timeout = full_timeout):
            try:
                data = await res
                ticker = data.get_name()
            except Exception as err:
                data = Error("Unknown", err)
                output.append(data)
                continue

            try:
                data = await data
            except Exception as err:
                data = Error(ticker, err)
            else:
                data = _format_information_result(data, ticker)
            output.append(data)

    return output


async def get_symbol_extra_data(symbols, timeout = 60, chunk = 20):
    
    symbols = list(always_iterable(symbols))
    output = []
    if not symbols:
        data = Error(symbols, ValueError("Parameter symbols cannot be empty"))
        output.append(data)

    if chunk > 50:
        data = Error(chunk, ValueError(f"Parameter chunk is too big: {chunk}"))
        output.append(data)

    if output:
        return output

    full_timeout = chunk * timeout
    if full_timeout <= 0:
        full_timeout = None

    if timeout <= 0:
        timeout = None

    for sym in chunked(sorted(symbols), chunk):
        tasks = [get_symbol_info(s, timeout) for s in sym]
        for res in asyncio.as_completed(tasks, timeout = full_timeout):
            try:
                data = await res
                ticker = data.get_name()
            except Exception as err:
                data = Error("Unknown", err)
                output.append(data)
                continue

            try:
                data = await data
            except Exception as err:
                data = Error(ticker, err)
            else:
                data = _format_information_extra_result(data, ticker)
            output.append(data)

    return output


async def get_price_group(
    symbols,
    start_date,
    end_date,
    interval = "1d",
    actions = True,
    timeout = 60,
    chunk = 20,
):

    symbols = list(always_iterable(symbols))
    output = []
    if not symbols:
        data = Error(symbols, ValueError("Parameter symbols cannot be empty"))
        output.append(data)

    if chunk > 50:
        data = Error(chunk, ValueError(f"Parameter chunk is too big: {chunk}"))
        output.append(data)

    try:
        bgn = parse_date(start_date).date()
    except Exception as err:
        data = Error(start_date, err)
        output.append(data)

    try:
        end = parse_date(end_date).date()
    except Exception as err:
        data = Error(start_date, err)
        output.append(data)

    if output:
        return output

    full_timeout = chunk * timeout
    if full_timeout <= 0:
        full_timeout = None

    if timeout <= 0:
        timeout = None

    for sym in chunked(sorted(symbols), chunk):
        tasks = [
            get_price(
                s,
                bgn,
                end + datetime.timedelta(days = 1),
                interval = interval,
                actions = actions,
                timeout = timeout
            ) for s in sym
        ]

        for res in asyncio.as_completed(tasks, timeout = full_timeout):
            try:
                data = await res
                ticker = data.get_name()
            except Exception as err:
                data = Error("Unknown", err)
                output.append(data)
                continue

            try:
                data = await data
            except Exception as err:
                data = Error(ticker, err)
            else:
                data = _format_price_result(data, ticker)
                if isinstance(data, Error):
                    output.append(data)
                    continue
                data = Success({ticker : data.data})
            output.append(data)

    e, s = partition(lambda x: isinstance(x, Success), output)
    s = dict(it.chain.from_iterable(val.data.items() for val in s))
    s = Success(s)
    output = [s] + list(e)
    return output


def _get_cashflow(ticker: yf.Ticker, freq = "yearly"):
    logger.info(f"Getting cashflow information for {ticker.ticker}")
    data = ticker.get_cashflow(freq = freq)
    logger.info(f"Getting cashflow information for {ticker.ticker} - complete")
    return data


def _get_income_stmt(ticker: yf.Ticker, freq = "yearly"):
    logger.info(f"Getting income statement information for {ticker.ticker}")
    data = ticker.get_income_stmt(freq = freq)
    logger.info(f"Getting income statement information for {ticker.ticker} - complete")
    return data


def _get_balance_sheet(ticker: yf.Ticker, freq = "yearly"):
    logger.info(f"Getting balance sheet information for {ticker.ticker}")
    data = ticker.get_balance_sheet(freq = freq)
    logger.info(f"Getting balance sheet information for {ticker.ticker} - complete")
    return data


async def get_cashflow(symbol, freq = "yearly", timeout = 60):
    ticker = yf.Ticker(symbol.upper())
    task = asyncio.wait_for(asyncio.to_thread(_get_cashflow, ticker, freq), timeout)
    task = asyncio.create_task(task, name = ticker.ticker)
    return task


async def get_income_stmt(symbol, freq = "yearly", timeout = 60):
    ticker = yf.Ticker(symbol.upper())
    task = asyncio.wait_for(asyncio.to_thread(_get_income_stmt, ticker, freq), timeout)
    task = asyncio.create_task(task, name = ticker.ticker)
    return task


async def get_balance_sheet(symbol, freq = "yearly", timeout = 60):
    ticker = yf.Ticker(symbol.upper())
    task = asyncio.wait_for(asyncio.to_thread(_get_balance_sheet, ticker, freq), timeout)
    task = asyncio.create_task(task, name = ticker.ticker)
    return task


async def get_financials(symbol: str, freq: str = "yearly", timeout: int = 60) -> dict:

    if symbol == "":
        raise ValueError("Symbol must not be empty")

    if freq not in ("yearly", "quarterly"):
        raise ValueError(f"Freq must be either 'yearly' or 'quarterly', it is: {freq}")

    if timeout <= 0:
        timeout = None

    output = dict()
    cashflow = get_cashflow(symbol, freq = freq, timeout = timeout)
    income = get_income_stmt(symbol, freq = freq, timeout = timeout)
    balance_sheet = get_balance_sheet(symbol, freq = freq, timeout = timeout)

    for key, fut in [("cashflow", cashflow), ("income_statement", income), ("balance_sheet", balance_sheet)]:
        data, ticker = await _await_for_future(fut)
        if not isinstance(data, Error):
            try:
                data = _format_financials(data, ticker)
            except Exception as err:
                data = Error(f"There was an error when formatting financials {key} for ticker {ticker}", err)
            else:
                data = Success(data)

        output[key] = data

    return output


async def _await_for_future(fut):
    ticker = None
    try:
        data = await fut
        ticker = data.get_name()
        data = await data
    except Exception as err:
        data = Error(ticker, err)

    return data, ticker


def _format_financials(fin, ticker):
    fin_dict = fin.to_dict(orient = "index")
    for item in fin_dict.keys():
        fin_dict[item] = {k.strftime("%Y-%m-%d") : (None if pd.isna(v) else v) for k, v in fin_dict[item].items()}

    return fin_dict

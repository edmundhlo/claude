import eodhd.eodhd as eod # This is a sample Python script.
import pandas as pd
from io import StringIO
import json
import zipfile
from pathlib import Path
import re
import csv

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
API_TOKEN = None

def get_symbols():
    # Use a breakpoint in the code line below to debug your script.
    client = eod.EODHDClient(api_token=API_TOKEN)
    json = client.exchange_symbols(exchange='US', delisted=0)
    df = pd.DataFrame(json)
    return df

def get_daily_prices(exchange:str, symbol:str, exchange_group:str=None):
    client = eod.EODHDClient(api_token=API_TOKEN)

    try:
        if exchange_group is None:
            txt = client.eod(symbol=symbol + '.' + exchange , fmt='csv')
        else:
            txt = client.eod(symbol=symbol + '.' + exchange_group, fmt='csv')
        df = pd.read_csv(StringIO(txt), dtype=str)
    except Exception as e:
        df = pd.DataFrame([{
            "exchange": exchange,
            "symbol": symbol,
            "error_message": str(e)
        }])
        df.to_csv('./eod/failure__' + exchange + '_' + symbol + '.csv', na_rep="", index=False)
    else:
        df.to_csv('./eod/success__' + exchange + '_' + symbol + '.csv', na_rep="", index=False)



def get_fundamentals(exchange:str, symbol:str, exchange_group:str=None):
    client = eod.EODHDClient(api_token=API_TOKEN)

    try:
        if exchange_group is None:
            txt = client.fundamentals(symbol=symbol + '.' + exchange )
        else:
            txt = client.fundamentals(symbol=symbol + '.' + exchange_group)
        json_str = json.dumps(txt)
    except Exception as e:
        txt = {
            "exchange": exchange,
            "symbol": symbol,
            "error_message": str(e)
        }
        json_str = json.dumps(txt)
        with open('./fundamentals/failure__' + exchange + '_' + symbol + '.json', "w") as f:
            f.write(json_str)
    else:
        with open('./fundamentals/success__' + exchange + '_' + symbol + '.json', "w") as f:
            f.write(json_str)


def iter_fundamentals(zip_path:str):
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not info.filename.startswith("fundamentals/"):
                continue
            if not info.filename.endswith(".json"):
                continue
            with zf.open(info) as f:
                yield info.filename, json.load(f)

def find_in_zip(zip_path, directory, filename):
    """
    Search for `filename` under `directory` inside the zip at `zip_path`.
    Returns the full path inside the archive if found, else None.
    """
    prefix = directory.rstrip("/") + "/"
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith("/" + filename):
                return name
            # Also match if file sits directly in the directory
            if name == prefix + filename:
                return name
    return None


def load_from_zip(zip_path, archive_path):
    """Load a file from a zip archive into a DataFrame based on extension."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(archive_path) as f:
            if archive_path.endswith(".csv"):
                return pd.read_csv(f)
            elif archive_path.endswith(".json"):
                return pd.read_json(f)
            elif archive_path.endswith((".xlsx", ".xls")):
                return pd.read_excel(f)
            else:
                raise ValueError(f"Unsupported file type: {archive_path}")

def rename_files(directory, rename_fn):
    for old_path in Path(directory).rglob("*"):
        if not old_path.is_file():
            continue
        new_name = rename_fn(old_path)
        new_path = old_path.with_name(new_name)  # keeps parent dir intact
        old_path.rename(new_path)
        yield old_path, new_path




def combine_csvs(directory, output_path):
    """
    Stream-combine all AAAA_BBBB.csv files in `directory` into one CSV.
    Adds 'exchange' and 'symbol' columns from the filename.
    """
    pattern = re.compile(r"^([^_]+)_([^_]+)\.csv$")
    paths = sorted(p for p in Path(directory).iterdir() if pattern.match(p.name))

    if not paths:
        raise ValueError(f"No matching CSV files found in {directory}")

    with open(output_path, "w", newline="") as out:
        writer = csv.writer(out)
        header_written = False

        for path in paths:
            m = pattern.match(path.name)
            exchange, symbol = m.group(1), m.group(2)

            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)

                if not header_written:
                    writer.writerow(header + ["exchange", "symbol"])
                    header_written = True

                for row in reader:
                    writer.writerow(row + [exchange, symbol])



def main():
    # symbols = get_symbols()
    # US_traded = symbols.query("Type=='Common Stock' and Exchange in ( 'NYSE', 'NASDAQ','NMFQS','NYSE MKT', 'BATS', 'NYSE ARCA','AMEX','US')")
    # for index, security in US_traded.iterrows():
    #     get_fundamentals(security['Exchange'], security['Code'], exchange_group='US')
    #     print(security['Exchange'], security['Code'])

    # c = 0
    # for name, data in iter_fundamentals("US_common_stocks.zip"):
    #     if data.get('General', {}).get('ISIN', {}) is not None:
    #         exchange = data.get('General', {}).get('Exchange', '')
    #         symbol = data.get('General', {}).get('Code', '')
    #         get_daily_prices(exchange, symbol, exchange_group='US')
    #         c += 1
    #         print(str(c))

    # def rename_fn(old_path):
    #     return old_path.name.removeprefix("success__")
    #
    # for old_path, new_path in rename_files('./eod', rename_fn):
    #     pass
    #combine_csvs('./eod', 'eod_all_symbols.csv')
import eodhd.eodhd as eod # This is a sample Python script.
import pandas as pd
from io import StringIO

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def get_symbols():
    # Use a breakpoint in the code line below to debug your script.
    client = eod.EODHDClient(api_token='5e201edf8165d5.36678401')
    json = client.exchange_symbols(exchange='US', delisted=1)
    df = pd.DataFrame(json)
    df.to_csv("symbols.csv", na_rep="", index=False)


def get_daily_prices(exchange:str, symbol:str, serial_number:int):
    client = eod.EODHDClient(api_token='5e201edf8165d5.36678401')
    try:
        txt = client.eod(symbol=symbol+'.US', fmt='csv')
        df = pd.read_csv(StringIO(txt), dtype=str)
    except Exception as e:
        df = pd.DataFrame([{
            "exchange": exchange,
            "symbol": symbol,
            "error_message": str(e)
        }])
        df.to_csv('failure__' + exchange + '.' + symbol + '.csv', na_rep="", index=False)
    else:
        df.to_csv('sucess__' + exchange + '.' + symbol + '.csv', na_rep="", index=False)


def main():
    get_daily_prices('NASDAQ', 'SasdTMP', 0)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

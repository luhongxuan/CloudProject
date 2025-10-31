from datetime import date
import io
from typing import Optional, Union

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
from typing import Dict, Iterator, Optional, Tuple, Union

import cloudscraper
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
DATE_FORMAT = "%Y-%m-%d"

def get_soup(driver, start_dt: Optional[Union[date, str]], end_dt: Optional[Union[date, str]]) -> BeautifulSoup:
    # get most recent standings if date not specified
    if((start_dt is None) or (end_dt is None)):
        print('Error: a date range needs to be specified')
        return None
    
    url = "http://www.baseball-reference.com/leagues/daily.cgi?user_team=&bust_cache=&type=p&lastndays=7&dates=fromandto&fromandto={}.{}&level=mlb&franch=&stat=&stat_value=0".format(start_dt, end_dt)
    driver.get(url)
    s = driver.page_source
    # a workaround to avoid beautiful soup applying the wrong encoding
    # s = str(s).encode()
    return BeautifulSoup(s, "html.parser")


def get_table(soup: BeautifulSoup) -> pd.DataFrame:
    table = soup.find(id="daily")
    raw_data = []
    headings = [th.get_text() for th in table.find("tr").find_all("th")][1:]
    headings.append("mlbID")
    raw_data.append(headings)
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        row_anchor = row.find("a")
        mlbid = row_anchor["href"].split("mlb_ID=")[-1] if row_anchor else pd.NA  # ID str or nan
        cols = [ele.text.strip() for ele in cols]
        cols.append(mlbid)
        raw_data.append([ele for ele in cols])
    data = pd.DataFrame(raw_data)
    data = data.rename(columns=data.iloc[0])
    data = data.reindex(data.index.drop(0))
    return data


def pitching_stats_range(driver, start_dt: Optional[str]=None, end_dt: Optional[str]=None) -> pd.DataFrame:
    """
    Get all pitching stats for a set time range. This can be the past week, the
    month of August, anything. Just supply the start and end date in YYYY-MM-DD
    format.
    """
    # ensure valid date strings, perform necessary processing for query
    start_dt_date, end_dt_date = sanitize_date_range(start_dt, end_dt)
    if start_dt_date.year < 2008:
        raise ValueError("Year must be 2008 or later")
    if end_dt_date.year < 2008:
        raise ValueError("Year must be 2008 or later")
    # retrieve html from baseball reference
    soup = get_soup(driver, start_dt_date, end_dt_date)
    table = get_table(soup)
    table = table.dropna(how='all') # drop if all columns are NA
    #fix some strange formatting for percentage columns
    table = table.replace('---%', np.nan)
    #make sure these are all numeric
    for column in ['Age', '#days', 'G', 'GS', 'W', 'L', 'SV', 'IP', 'H',
                    'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'ERA', 'AB', '2B',
                    '3B', 'IBB', 'GDP', 'SF', 'SB', 'CS', 'PO', 'BF', 'Pit',
                    'WHIP', 'BAbip', 'SO9', 'SO/W']:
        table[column] = pd.to_numeric(table[column])
    #convert str(xx%) values to float(0.XX) decimal values
    for column in ['Str', 'StL', 'StS', 'GB/FB', 'LD', 'PU']:
        table[column] = table[column].replace('%','',regex=True).astype('float')/100

    table = table.drop('', axis=1)
    return table

def sanitize_date_range(start_dt: Optional[str], end_dt: Optional[str]) -> Tuple[date, date]:
	# If no dates are supplied, assume they want yesterday's data
	# send a warning in case they wanted to specify
	if start_dt is None and end_dt is None:
		today = date.today()
		start_dt = str(today - timedelta(1))
		end_dt = str(today)

		print('start_dt', start_dt)
		print('end_dt', end_dt)

		print("Warning: no date range supplied, assuming yesterday's date.")

	# If only one date is supplied, assume they only want that day's stats
	# query in this case is from date 1 to date 1
	if start_dt is None:
		start_dt = end_dt
	if end_dt is None:
		end_dt = start_dt

	start_dt_date = validate_datestring(start_dt)
	end_dt_date = validate_datestring(end_dt)

	# If end date occurs before start date, swap them
	if end_dt_date < start_dt_date:
		start_dt_date, end_dt_date = end_dt_date, start_dt_date

	# Now that both dates are not None, make sure they are valid date strings
	return start_dt_date, end_dt_date

def validate_datestring(date_text: Optional[str]) -> date:
	try:
		assert date_text
		return datetime.strptime(date_text, DATE_FORMAT).date()
	except (AssertionError, ValueError) as ex:
		raise ValueError("Incorrect data format, should be YYYY-MM-DD") from ex
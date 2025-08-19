import pandas as pd
from datetime import datetime, timedelta
from wrappers.time_it import timeit

def calculate_holidays(year):
    """
    Generate a dictionary of holiday dates for a given year.
    """
    holidays = {
        'Christmas Eve': datetime(year, 12, 24),
        'Christmas Day': datetime(year, 12, 25),
        'Boxing Day': datetime(year, 12, 26),
        'New Year\'s Eve': datetime(year, 12, 31),
        'New Year\'s Day': datetime(year, 1, 1),
        "Martin Luther King's Day": pd.Timestamp(f"{year}-01-01") + pd.DateOffset(weekday=2) + pd.DateOffset(weekday=1, weeks=2),  # Third Monday of January
        'Good Friday': calculate_good_friday(year),
        'Easter Monday': calculate_easter_monday(year),
        'Thanksgiving': calculate_thanksgiving(year),
    }
    return {date: name for name, date in holidays.items()}

def calculate_good_friday(year):
    """Calculate Good Friday based on Easter Sunday."""
    easter = calculate_easter(year)
    return easter - timedelta(days=2)

def calculate_easter(year):
    """Calculate Easter Sunday date using a simple algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day)

def calculate_easter_monday(year):
    """Calculate Easter Monday based on Easter Sunday."""
    return calculate_easter(year) + timedelta(days=1)

def calculate_thanksgiving(year):
    """Calculate Thanksgiving Day (fourth Thursday of November)."""
    return pd.Timestamp(f"{year}-11-01") + pd.DateOffset(weekday=3, weeks=3)

@timeit
def create_time_features(df):
    df['Time'] = df['Time'].apply(pd.to_datetime, errors='coerce', utc=True)
    df['time_idx'] = ((df['Time'] - df['Time'].min()).dt.total_seconds() // (15 * 60)).astype(int)

    # Convert to string before categorizing to avoid numeric category issue
    df['hour'] = df['Time'].dt.hour.astype(str).astype('category')
    df['day_of_week'] = df['Time'].dt.dayofweek.astype(str).astype('category')
    df['month'] = df['Time'].dt.month.astype(str).astype('category')

    # Define trading sessions
    df['session'] = df['hour'].apply(lambda x: 'Sydney' if 17 <= int(x) <= 19 else 'Tokyo/Sydney' if 19 <= int(x) or int(x) <= 2  else 'Tokyo' if 2 <= int(x) < 3 else 'Tokyo/London' if 3 <= int(x) < 4 else 'London' if 4 <= int(x) < 8 else "London/New York" if 8 <= int(x) < 12 else "New York" if 12 <= int(x) < 17 else "None").astype('category')

    holidays = {}
    for year in df['Time'].dt.year.unique():
        holidays.update(calculate_holidays(year))

    # Assign holiday or event categories
    df['holiday'] = df['Time'].apply(lambda x: holidays.get(x.date(), 'Non-Holiday')).astype('category')
    df['single_group'] = 'EURUSD' # Necessary for TFT PyTorch Library

    return df
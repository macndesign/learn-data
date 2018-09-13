import sys
import pandas.util.testing as tm
import numpy as np
import pandas as pd
from itertools import product

tm.N, tm.K = 15, 3  # Module-level default rows/columns
np.random.seed(444)
df = tm.makeTimeDataFrame(freq='M').head()
print(df)

df = tm.makeTimeDataFrame().head()
print(df)

makers = [i for i in dir(tm) if i.startswith('make')]
print(makers, len(makers))

# Accessors
print(pd.Series._accessors)

# str accessor
addr = pd.Series([
    'Washington, D.C. 20003',
    'Brooklyn, NY 11211-1755',
    'Omaha, NE 68154',
    'Pittsburgh, PA 15211'
])

print(addr.str.upper())
print(addr.str.count(r'\d'))

regex = (r'(?P<city>[A-Za-z ]+), '      # One or more letters
         r'(?P<state>[A-Z]{2}) '        # 2 capital letters
         r'(?P<zip>\d{5}(?:-\d{4})?)')  # Optional 4-digit extension

print(addr.str.replace('.', '').str.extract(regex))

# dt accessor
daterng = pd.Series(pd.date_range('2017', periods=9, freq='Q'))
print(daterng)
print(daterng.dt.day_name())

# Second-half of year only
print(daterng[daterng.dt.quarter > 2])

print(daterng[daterng.dt.is_year_end])

# Datetimeindex
datecols = ['year', 'month', 'day']
df = pd.DataFrame(list(product([2017, 2016], [1, 2], [1, 2, 3])), columns=datecols)
df['data'] = np.random.randn(len(df))
print(df)
df.index = pd.to_datetime(df[datecols])
print(df)
df = df.drop(datecols, axis=1).squeeze()
print(df.head())
print(df.index.dtype_str)

colors = pd.Series([
    'periwinkle',
    'mint green',
    'burnt orange',
    'periwinkle',
    'burnt orange',
    'rose',
    'rose',
    'mint green',
    'rose',
    'navy'
])

print(colors.apply(sys.getsizeof))
mapper = {v: k for k, v in enumerate(colors.unique())}
print(mapper)
# as int
as_int = colors.map(mapper)
print(as_int)
print(as_int.apply(sys.getsizeof))
print(pd.factorize(colors)[0])

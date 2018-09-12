import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creating a Series by passing a list of values, letting pandas create a default integer index:
s = pd.Series([1,3,5,np,None,6,8])
print(s)

# Creating a DataFrame by passing a NumPy array, with a datetime index and labeled columns:
dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

# Creating a DataFrame by passing a dict of objects that can be converted to series-like.
df2 = pd.DataFrame({
    'A': 1.,
    'B': pd.Timestamp('20130102'),
    'C': pd.Series(1, index=list(range(4)), dtype='float'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(['test', 'train', 'test', 'train']),
    'F': 'foo'
})
print(df2)

# The columns of the resulting DataFrame have different dtypes.
print(df2.dtypes)

# Here is how to view the top and bottom rows of the frame:
print(df.head)
print(df.tail(3))

# Display the index, columns, and the underlying NumPy data:
print(df.index)
print(df.columns)
print(df.values)

# describe() shows a quick statistic summary of your data:
print(df.describe())

# Transposing your data:
print(df.T)

# Sorting by an axis:
print(df.sort_index(axis=1, ascending=False))

# Sorting by values
print(df.sort_values(by='B'))

# Selecting
print(df['A'])
print(df[0:3])
print(df['20130102':'20130104'])

# getting cross selection using a label
print(df.loc[dates[0]])

# selecting on a multiple axis by label
print(df.loc[:,['A', 'B']])

# showing label slicing, both endpoints are included
print(df.loc['20130102':'20130104', ['A', 'B']])

# reduction in the dimensions of the returned object
print(df.loc['20130102', ['A', 'B']])

# for getting a scalar value
print(df.loc[dates[0], 'A'])

# select via the position of the passed integers
print(df.iloc[3])

# By integer slices, acting similar to numpy/python:
print(df.iloc[3:5,0:2])

# By lists of integer position locations, similar to the numpy/python style:
print(df.iloc[[1,2,4],[0,2]])

# For slicing rows explicitly:
print(df.iloc[1:3,:])

# For slicing columns explicitly:
print(df.iloc[:,1:3])

# For getting a value explicitly:
print(df.iloc[1,1])

# For getting fast access to a scalar (equivalent to the prior method):
print(df.iat[1,1])

# Boolean indexing
print(df[df.A > 0])

# Selecting values from a DataFrame where a boolean condition is met.
print(df[df > 0])

# Using the isin() method for filtering:
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
print(df2)
print(df2[df2['E'].isin(['two', 'four'])])

# Setting a new column automatically aligns the data by the indexes.
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
print(s1)
df['F'] = s1
print(df)

# Setting values by label
df.at[dates[0], 'A'] = 0
print(df)

# Setting values by position
df.iat[0, 1] = 0
print(df)

# Setting by assigning with a NumPy array:
df.loc[:, 'D'] = np.array([5] * len(df))
print(df)

# A where operation with setting
df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)

# Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data.
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
print(df1)

# To drop any rows that have missing data
print(df1.dropna(how='any'))

# Filling missing data
print(df1.fillna(value=5))

# To get boolean mask where values are nan
print(pd.isna(df1))

# performing a descriptive statistic
print(df.mean())

# Same operation on the other axis
print(df.mean(1))

# Operating with objects that have different dimensionality and need alignment. In addition, pandas automatically broadcasts along the specified dimension.
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
print(s)
print(df.sub(s, axis='index'))

# Applying function to the data
print(df.apply(np.cumsum))
print(df.apply(lambda x: x.max() - x.min()))

# See more at Histogramming and Discretization
s = pd.Series(np.random.randint(0, 7, size=10))
print(s)
print(s.value_counts())

# String methods
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(s.str.lower())

# Concat data frame
df = pd.DataFrame(np.random.randn(10, 4))
print(df)
pieces = [df[:3], df[3:7], df[7:]]
print(pd.concat(pieces))

# Join
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
print(left)
print(right)
print(pd.merge(left, right, on='key'))

# Another Join
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
print(left)
print(right)
print(pd.merge(left, right, on='key'))

# Append rows to a DataFrame
df = pd.DataFrame(np.random.randn(8, 4), columns=list('ABCD'))
print(df)
s = df.iloc[3]
print(s)
print(df.append(s, ignore_index=True))

# Grouping
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})
print(df)
print(df.groupby('A').sum())
print(df.groupby(['A', 'B']).sum())

# Reshaping
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
print(df)
df2 = df[:4]
print(df2)
stacked = df2.stack()
print(stacked)

# Pivot table
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),
                   'E' : np.random.randn(12)})

print(df)
print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))

# Time Series
rng = pd.date_range('1/1/2012', periods=100, freq='S')
print(rng)
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts)
print(ts.resample('5Min').sum())

# Timezone representations
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
print(rng)
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)
ts_utc = ts.tz_localize('UTC')
print(ts_utc)

# Converting to another time zone
print(ts_utc.tz_convert('US/Eastern'))

# Converting between time span representations
rng = pd.date_range('1/1/2012', periods=5, freq='M')
print(rng)
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)
ps = ts.to_period()
print(ps)
print(ps.to_timestamp())

prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
print(ts.head())

# Categoricals
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
print(df)
df['grade'] = df['raw_grade'].astype('category')
print(df['grade'])

# Rename categories
df['grade'].cat.categories = ['very good', 'good', 'very bad']
print(df['grade'])

# Sorting is per order in the categories, not lexical order.
print(df.sort_values(by='grade'))

# Grouping by a categorical column also shows empty categories
print(df.groupby('grade').size())

# Ploting
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
# ts.plot()
# plt.show()

# On a DataFrame, the plot() method is a convenience to plot all of the columns with labels:
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()
# plt.figure()
df.plot()
plt.legend(loc='best')
plt.show()

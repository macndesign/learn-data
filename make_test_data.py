import pandas.util.testing as tm
import numpy as np

tm.N, tm.K = 15, 3  # Module-level default rows/columns
np.random.seed(444)
df = tm.makeTimeDataFrame(freq='M').head()
print(df)

df = tm.makeTimeDataFrame().head()
print(df)

makers = [i for i in dir(tm) if i.startswith('make')]
print(makers, len(makers))

df = tm.makeStringSeries()
print(df)

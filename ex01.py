import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
irisdf = pd.read_csv('iris.csv')
customerdf = pd.read_csv('customers.csv')

if __name__ == '__main__':
    print(customerdf)
    print(irisdf)
    print('length:', len(irisdf)) # length of data set
    print('shape:', irisdf.shape) # length and width of data set
    print('size:', irisdf.size) # length * width
    print('min:', irisdf['sepal_width'].min())
    print('max:', irisdf['sepal_width'].max())
    print('mean:', irisdf['sepal_width'].mean())
    print('median:', irisdf['sepal_width'].median())
    print('50th percentile:', irisdf['sepal_width'].quantile(0.5)) # 50th percentile, also known as median
    print('5th percentile:', irisdf['sepal_width'].quantile(0.05))
    print('10th percentile:', irisdf['sepal_width'].quantile(0.1))
    print('95th percentile:', irisdf['sepal_width'].quantile(0.95))
    print(customerdf['industry'].value_counts())
    print(customerdf['industry'].value_counts(ascending=True))
    timedf = pd.DataFrame(np.random.randn(1000, 4), index=pd.date_range('1/1/2015', periods=1000), columns=list('ABCD'))
    timedf = timedf.cumsum()
    # timedf.plot()
    # plt.hist(irisdf['sepal_length'])
    # plt.show()
    fig, ax = plt.subplots()
    ax.scatter(irisdf['sepal_length'], irisdf['sepal_width'], color='green')
    ax.set(
        xlabel='length',
        ylabel='width',
        title='Iris sepal sizes'
    )
    plt.show()

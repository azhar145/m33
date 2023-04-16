# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import pandas as pd
import datetime as dt
import sys
import Technicals
from Technicals import Techinicalsbb

#############
##data = np.loadtxt("example.txt", delimiter=",")
global ticker
ticker='tsla'

df = yf.download(ticker, period='300d', interval='1d',prepost = False)
df=pd.DataFrame(df)
##df=df['Close']
##df.index = pd.to_datetime(df.index)
##print(df['Close'].index)

dfp=df
print(df.columns)
df=df[['Close']]


fig, axes = plt.subplots(ncols=1)
plt.xticks(rotation=45, ha="right", rotation_mode="anchor") #rotate the x-axis values
plt.subplots_adjust(bottom = 0.2, top = 0.9) #ensuring the dates (on the x-axis) fit in the screen
plt.ylabel('Stock')
plt.xlabel('Dates')
plt.grid(True)
plt.ylim((df['Close'].min(),df['Close'].max()))

dfpq=dfp[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
dfp=dfp[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
dfp.columns=['open', 'high', 'low', 'close', 'adj Close', 'volume']

dfpp=Technicals.Techinicalsbb(dfp)
##print(dfpp.columns)
dfp=dfpp[['close','vwap','EMA_50','EMA_10','EMA_21','SAR','EMA_100','EMA_200']]
##print(dfp)


##
dfp2=dfpp[['ROC']]
##dfp2.columns=['open', 'high', 'low', 'close', 'adj Close', 'volume']
##dfp2.reset_index()
##dfp2.set_index('Date')
##
##dfp2=Technicals.Techinicalsbb(dfp2)
##print(dfp2.columns)
##dfp2=dfp2[['close','vwap','EMA_50','EMA_10','EMA_21','SAR','EMA_100','EMA_200']]
##print(dfp2)



##df_interest = df.loc[
##    df['Country/Region'].isin(['United Kingdom', 'US', 'Italy', 'Germany'])


##s=dfp.columns.get_loc('vwap')
##i=dfp.columns.get_loc('close')
print('trash 7')
def buildmebarchart(i=int):
    import matplotlib.animation as animation
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
 


    
##    i=dfp.columns.get_loc('close')

##    p = plt.plot(dfp[:i].index, dfp[:i].values,'o-', lw = 2,color='b') #note it only returns the dataset, up to the point i
    p2 = plt.plot(dfp[:i].index, dfp['close'][:i].values,'-', lw = 1,color='red',label='close') #note it only returns the dataset, up to the point i
    p3 = plt.plot(dfp[:i].index, dfp['vwap'][:i].values,'o', lw = 1,color='cyan',label='vwap')
    p4 = plt.plot(dfp[:i].index, dfp['EMA_50'][:i].values,'--', lw = 0.8,color='pink',label='EMA_50')
    p4a = plt.plot(dfp[:i].index, dfp['EMA_100'][:i].values,'--', lw = 1.2,color='olive',label='EMA_100')
    p4a = plt.plot(dfp[:i].index, dfp['EMA_200'][:i].values,'--', lw = 3,color='grey',label='EMA_200')
    p5 = plt.plot(dfp[:i].index, dfp['EMA_10'][:i].values,'-', lw = 1.5,color='green',label='EMA_10')
    p6 = plt.plot(dfp[:i].index, dfp['EMA_21'][:i].values,'-', lw = 1.4,color='yellow',label='EMA_21')
    p7 = plt.plot(dfp[:i].index, dfp['SAR'][:i].values,'-', lw = .4,color='blue',label='SAR')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title=ticker




    g2 = plt.plot(dfp2[:i].index, dfp2['ROC'][:i].values,'-', lw = 1,color='r',label='ROC')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = dict(zip(labels, handles))
##    plt.legend(by_label.values(), by_label.keys())

    
    
##    for i in range(0,4):
##        p[i].set_color(color[i]) #set the colour of each curve

import matplotlib.animation as ani
animator = ani.FuncAnimation(fig, buildmebarchart, interval = 0)

plt.show()

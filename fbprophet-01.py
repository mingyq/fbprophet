#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @date: 2021/5/7 17:52
# @name: fbprophet-01
# @author：mmm

# 文章参考
# https://blog.csdn.net/qq_23860475/article/details/81354467
# https://facebook.github.io/prophet/docs/quick_start.html

import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt
from fbprophet.plot import plot_plotly, plot_components_plotly

# df = pd.read_csv('example_wp_log_peyton_manning.csv')
df = pd.read_csv('water flosser.csv')
# print(df.head())
# 包含节假日期


m = Prophet()
m.add_country_holidays(country_name='US')
m.fit(df)
print(m.train_holiday_names)
future = m.make_future_dataframe(periods=6, freq='M')
forecast = m.predict(future)
fig = m.plot_components(forecast)
plt.show()


# future = m.make_future_dataframe(periods=365, freq='D')
# future = m.make_future_dataframe(periods=10, freq='M')
# 延伸到未来的日期(H,D,M;天、月份的最后一天);可包含当前的日期
# print(future.tail(10))
# print(future.head())
# print(type(future), future.columns)
# forecast = m.predict(future)
# col_list = forecast.columns.tolist()
# print(col_list)
# print(forecast[['ds', 'trend']].head())
# print(forecast.tail())
# print(type(forecast))
# # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# fig1 = m.plot(forecast)
# plt.show()
# plot_plotly(m, forecast)

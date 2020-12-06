import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
from datetime import datetime

ROOT_PATH = os.path.dirname(os.path.realpath(__file__ + '/../')) + '/'
FEATURES_PATH = ROOT_PATH + 'features/'
INPUT_PATH = ROOT_PATH + 'input/'
SUBM_PATH = ROOT_PATH + 'subm/'
SUBM_PATH_DETAILED = SUBM_PATH + 'detailed/'
MODELS_PATH = ROOT_PATH + 'models/'

df = pd.read_csv(SUBM_PATH + 'rus_regions_train.csv')
df = df.sort_values(by=['date'])
# df = df[df['shift_day'] == 1]

df_moscow = df[['name1', 'target', 'pred', 'date']]
df_moscow = df_moscow[df_moscow['name1'] == 'Moscow__Russia']
dates = [datetime.strptime(date, '%Y.%m.%d') for date in df_moscow['date'].to_numpy()]


fig, ax = plt.subplots()
ax.plot(dates, df_moscow['target'].to_numpy(), color='b', label = 'Actual')
ax.plot(dates, df_moscow['pred'].to_numpy(), color='r', label = 'Predicted')

years = mdates.MonthLocator()
years_fmt = mdates.DateFormatter('%d.%m.%Y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
fig.autofmt_xdate()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate()  # Rotation
plt.legend()
plt.show()

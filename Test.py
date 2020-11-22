import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model

df1 = pd.read_csv('/Users/dattatreya/Desktop/analytics_assignment_data.csv')
print(df1.head(5))
print(df1.dtypes)

df = df1[(df1['order_status']=='Cancelled') & (df1['service']!='GO-TIX')]

df['DateTime'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['DateTime'].dt.weekday_name
df['day'] = df['DateTime'].dt.day
df['month'] = df['DateTime'].dt.month
df['dow']=df['DateTime'].dt.dayofweek +1
df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x=='Saturday' or x=='Sunday' else 0)
print(df.head(5))

df['day'].fillna(0, inplace = True)
df['month'].fillna(0, inplace = True)
df['dow'].fillna(0, inplace = True)
df['weekend'].fillna(0, inplace = True)

data = {}
data1 = {}

column_values = df[["service"]].values
unique_values =  np.unique(column_values)
print(unique_values)

j=0
for i in unique_values:
    #print(i)

    data[j] = df[(df['service']==i) & (df['date']<='2020-3-29')]
    data1[j] = df[(df['service']==i) & (df['date']>'2020-3-29')]
    j+=1

#print(data[0])

# d1 = data[0][data[0]['date']<='2016-03-29']
d_1 = data[0][data[0]['date']>'2016-03-29']
# d2 = data[1][data[1]['date']<='2016-03-29']
d_2 = data[1][data[1]['date']>'2016-03-29']
# d3 = data[2][data[2]['date']<='2016-03-29']
d_3 = data[2][data[2]['date']>'2016-03-29']
# d4 = data[3][data[3]['date']<='2016-03-29']
d_4 = data[3][data[3]['date']>'2016-03-29']
# d5 = data[4][data[4]['date']<='2016-03-29']
d_5 = data[4][data[4]['date']>'2016-03-29']
# d6 = data[5][data[5]['date']<='2016-03-29']
d_6 = data[5][data[5]['date']>'2016-03-29']
# d7 = data[6][data[6]['date']<='2016-03-29']
d_7 = data[6][data[6]['date']>'2016-03-29']
# d8 = data[7][data[7]['date']<='2016-03-29']
d_8 = data[7][data[7]['date']>'2016-03-29']
# d9 = data[8][data[8]['date']<='2016-03-29']
d_9 = data[8][data[8]['date']>'2016-03-29']
# d10 = data[9][data[9]['date']<='2016-03-29']
d_10 = data[9][data[9]['date']>'2016-03-29']


def MAPE(predict,target):
    return ( abs((target - predict) / target).mean()) * 100

print(data[0][['day','month','dow','weekend']].head())
print(d_1[['day','month','dow','weekend']].head())

regr1 = linear_model.LinearRegression()
regr1.fit(data[0][['day','month','dow','weekend']], data[0][['total_cbv']])
y_pred1 = regr1.predict(d_1[['day','month','dow','weekend']])
m1 = MAPE(y_pred1,d_1[['total_cbv']])
print(m1)

regr1 = linear_model.LinearRegression()
regr1.fit(data[1][['day','month','dow','weekend']], data[1][['total_cbv']])
y_pred1 = regr1.predict(d_2[['day','month','dow','weekend']])
m2 = MAPE(y_pred1,d_2[['total_cbv']])

regr1 = linear_model.LinearRegression()
regr1.fit(data[2][['day','month','dow','weekend']], data[2][['total_cbv']])
y_pred1 = regr1.predict(d_3[['day','month','dow','weekend']])
m3 = MAPE(y_pred1,d_3[['total_cbv']])

regr1 = linear_model.LinearRegression()
regr1.fit(data[3][['day','month','dow','weekend']], data[3][['total_cbv']])
y_pred1 = regr1.predict(d_4[['day','month','dow','weekend']])
m4 = MAPE(y_pred1,d_4[['total_cbv']])

regr1 = linear_model.LinearRegression()
regr1.fit(data[4][['day','month','dow','weekend']], data[4][['total_cbv']])
y_pred1 = regr1.predict(d_5[['day','month','dow','weekend']])
m5 = MAPE(y_pred1,d_5[['total_cbv']])

regr1 = linear_model.LinearRegression()
regr1.fit(data[5][['day','month','dow','weekend']], data[5][['total_cbv']])
y_pred1 = regr1.predict(d_6[['day','month','dow','weekend']])
m6 = MAPE(y_pred1,d_6[['total_cbv']])

regr1 = linear_model.LinearRegression()
regr1.fit(data[6][['day','month','dow','weekend']], data[6][['total_cbv']])
y_pred1 = regr1.predict(d_7[['day','month','dow','weekend']])
m7 = MAPE(y_pred1,d_7[['total_cbv']])

regr1 = linear_model.LinearRegression()
regr1.fit(data[7][['day','month','dow','weekend']], data[7][['total_cbv']])
y_pred1 = regr1.predict(d_8[['day','month','dow','weekend']])
m8 = MAPE(y_pred1,d_8[['total_cbv']])

regr1 = linear_model.LinearRegression()
regr1.fit(data[8][['day','month','dow','weekend']], data[8][['total_cbv']])
y_pred1 = regr1.predict(d_9[['day','month','dow','weekend']])
m9 = MAPE(y_pred1,d_9[['total_cbv']])

regr1 = linear_model.LinearRegression()
regr1.fit(data[9][['day','month','dow','weekend']], data[9][['total_cbv']])
y_pred1 = regr1.predict(d_10[['day','month','dow','weekend']])
m10 = MAPE(y_pred1,d_10[['total_cbv']])


print(m2)
a = np.array([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10])
print(sorted(a, reverse = True))











#print(MAPE(y_pred1,d_1[['total_cbv']]))

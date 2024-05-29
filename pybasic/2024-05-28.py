#%%
# EDA 탐색적 데이터 분석 할거임 ㅎㅎ
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
# %%

from sklearn.datasets import load_iris
iris=load_iris()
print(iris)
print(type(iris),iris)
# %%
iris.keys()
# %%
print(iris['data'].shape)
# %%
df=pd.DataFrame(iris['data'])
df
# %%
print(iris['feature_names'])
# %%
cols=['s1','sw','pl','pw']
df=pd.DataFrame(iris['data'],columns=cols)
df

# %%
# 딕셔너리 설명하기 위해 아래 내용 실행
cap={'한국':['서울','경기도','부산'],'일본':'도쿄'}
print(type(cap),cap.keys(),cap.values())
# %%
print(iris['target'].shape)
# %%
print(iris['target_names'])
# %%
iris.keys()
print(iris['DESCR'])
# %%
# 대표값을 통한 EDA  (여러 수치의 요약본 카운트, 평균값, 최소값, 최대값 등등)
df.describe()
# %%
#csv 로 출력
df.to_csv('./iris.csv')
# %%
# 그래프 그려주는 녀석 matplotlib
plt.plot(df)
# %%
# 위랑 똑같으나 이건 범례까지 있음 
df.plot()
# %%
df.plot(kind='line')

# %%
df.plot(kind='kde')
# %%
# 산점도  봐보겠음
df.plot(kind='scatter',x='s1',y='sw')
# %%
#box plot 
df.plot(kind='box', vert=False)
# %%
import seaborn as sns
sns.boxplot(df)
# %%
df
# %%
#라벨이라는 새열을 생성 거기에 타겟을 넣음
df['label']=iris['target']
df
# %%
sns.pairplot(df,hue='label')
# %%

df.values

# %%
#형태보기
df.shape
# %%
#기초 통계자료
df.describe()
# %%
df.dtypes
# %%
df.head()

# %%
#데이터 순서 편향 
df.tail()
# %%
#보시는 바와 같이 라벨이 반복되고 있어서 데이터의 순서적 편향이 예상됨
#따라서 머신러닝의 경우 셔플하는 것이 필요할것으로 보여진다.

df[df.duplicated(keep=False)]
# %%
#중복값이 있는것으로 확인

df=df.drop_duplicates()
df.shape
# %%
df.isna().sum()
df=df.dropna()
df.shape
# 빈값은 없었음
# %%
# 시각화 
df['s1'].plot(kind='hist',bins=30)
# %%
# 머신러닝 딥러닝은 정규화가 딱히 필요없음
# 최적의 값을 찾아야 해서
# 크게 정규성이 없어 보인다.
df.plot(kind='scatter',x='s1',y='sw')

# %%
sns.pairplot(df,hue='label')
# %%
# 시각적으로 어느정도 분리될것으로 보여지나
# 겹치는 부분이 많아서 통계적 분리는 어려울것으로 보여진다.
# (복선)왜 나는 머신러닝을 써야하나?
# 우상향 하는 그래프로 볼때 상관성 분석이 필요할것으로 보여진다.

sns.heatmap(df.iloc[:,0:4].corr(),annot=True)
# %%
# pw와 pl의 상관성이....있는것 같음
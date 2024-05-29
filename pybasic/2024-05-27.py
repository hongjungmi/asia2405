#%%
# #%% shift + enter
print('hello world')
# %%
print("홍정민짱") 
# %%
# 변수 명명 규칙
a=1
print(a)
# %%
print(print)
# %%
#리스트
a=[1,'1',[2,3,'시']]
print(a,'타입:',type(a[0]),type(a[1]))
a[0]=5  # 수정쌉가능
print(a)
# %%
#튜플
a=[1,'1',[2,3,'시']]
print(a,'타입:',type(a[0]),type(a[1]))
a[0]=5  # 수정쌉불가
print(a)
# %%
#딕셔너리
b={'한국':'서울','일본':'도쿄'}
print(b,type(b),b['한국'])# 접근은 키값을 넣어서 접근


# %%
#집합
c=set([1,1,2,2,3,4,5])
print(c,type(c))
# %%
import numpy as np
# 동일 데이터 타입으로 변환하여 사용
a=[1,2,3,'4']
ar=np.array(a).astype(int)
print(ar,type(ar))
# %%

a=range(12)
ar=np.array(a)
print(a,type(a),ar,type(ar))
print(range(10))
for i in range(10):
    print(i)
# %%
#형태 변환(차원변환)이 자유로움
print(ar,'형태:',ar.shape)
ar34=ar.reshape(3,4)
print(ar34,'차원:',ar34.shape)

# %%
#슬라이싱
ar34[1]
# %%
# 연산

# %%

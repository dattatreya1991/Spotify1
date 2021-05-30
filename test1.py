import math

def solution(lst, X):
    # Write your code here.
    n = len(lst)
    p = n * X / 100
    print(p)
    print(n)
    if p.is_integer():
        return sorted(lst)[int(p)]
    else:
        return sorted(lst)[int(math.ceil(p)) - 1]



lst=[1,3,5,1,3]
X=0.1
solution(lst,X)

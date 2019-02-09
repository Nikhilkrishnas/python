a = [int(x) for x in input().split()]
print("before sorting")
print(a)
l = len(a)
for i in range(l):
    for j in range(l):
        if a[i] < a[j]:
            temp = a[i]
            a[i] = a[j]
            a[j]= temp
print("after sorting")
print(a)
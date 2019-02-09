
items = []
i = int(input("enter a starting value"))
n = int(input("enter an ending value"))
for i in range(i, n):
    s = str(i)
    if (int(s[0])%2!=0) and (int(s[1])%2!=0) and (int(s[2])%2!=0):
        items.append(s)
print(items)

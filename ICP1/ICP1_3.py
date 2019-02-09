# ICP1 Part 3

string1 = input("Enter a string: ")

n = sum(c.isdigit() for c in string1)
l = sum(c.isalpha() for c in string1)
s = sum(c.isspace() for c in string1)

print('Numbers: ',n,' Letters: ',l,' Words: ',s+1)
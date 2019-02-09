# ICP1 Part2

num1str = input('Enter a number: ')
num1 = int(num1str)
number = num1
sum = 0
rem = 0
while True:
    if num1 != 0:
        rem = num1%10
        sum = sum*10+rem
        num1 = int(num1/10)

    else:
        break

print('the reverse of ',num1str,' is ',sum)

num1String = input('Please enter an integer: ')
num2String = input('Please enter a second integer: ')

num1 = int(num1String)
num2 = int(num2String)

print('Addition of ',num1,' and ',num2,' is ',num1+num2)
print('Subtraction of ',num1,' and ',num2,' is ',num1-num2)
print('Multiplication of ',num1,' and ',num2,' is ',num1*num2)
print('Division of ',num1,' and ',num2,' is ',num1/num2)
print('Modulus of ',num1,' and ',num2,' is ',num1%num2)
print('Floor Division of ',num1,' and ',num2,' is ',num1//num2)
print('Exponent of ',num1,' and ',num2,' is ',num1**num2)
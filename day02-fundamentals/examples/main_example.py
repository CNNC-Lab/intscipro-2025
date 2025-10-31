print("\nMethod 2: Tuple unpacking (Pythonic way)")
a = 10
b = 20
print(f"Before swap: a={a}, b={b}")
a, b = b, a
print(f"After swap: a={a}, b={b}")


c = 100
d = 200
print(f"Before swap: c={c}, d={d}")
c, d = d, c
print(f"After swap: c={c}, d={d}")


def swap_var(x, y):
    print(f"Before swap: x={x}, y={y}")
    x, y = y, x
    print(f"After swap: x={x}, y={y}")
    return y, x

a, b = swap_var(a, c)


#################################
from auxiliary import swap_var

a, b = swap_var(a, c)




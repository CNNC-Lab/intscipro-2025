def swap_var(x, y):
    print(f"Before swap: x={x}, y={y}")
    x, y = y, x
    print(f"After swap: x={x}, y={y}")
    return y, x
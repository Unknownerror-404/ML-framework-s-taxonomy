table =[[0.0000, 0.0961, 0.1368, 0.0250, 0.1043, -0.1149, 0.1750], 
        [-0.0961, 0.0000, 0.0365, -0.0726, 0.0101, -0.2096, 0.0726],
        [-0.1368, -0.0365, 0.0000, -0.1129, -0.0254, -0.2554, 0.0418],
        [-0.0250, 0.0726, 0.1129, 0.0000, 0.0814, -0.1417, 0.1520],
        [-0.1043, -0.0101, 0.0254, -0.0814, 0.0000, -0.2153, 0.0646],
        [0.1149, 0.2096, 0.2554, 0.1417, 0.2153, 0.0000, 0.2912],
        [-0.1750, -0.0762, -0.0418, -0.1520, -0.0646, -0.2912, 0.0000]]

row = len(table)
column = len(table[0])

maximum = -float("inf")
minimum = float("inf")
m = n = p = q = 0
for i in range(row):
    for j in range(column):
        if table[i][j] > maximum:
            maximum = table[i][j]
            m, n = i, j
        if table[i][j] < minimum:
            minimum = table[i][j]
            p, q = i, j

print(f"Maximum value in d is: {maximum} located at index ({m}, {n})")
print(f"Minimum value in d is: {minimum} located at index ({p}, {q})")
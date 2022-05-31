import fractions as f


h = 1/4
x = [0, f.Fraction(1, 4), f.Fraction(1, 2), f.Fraction(3, 4), 1]
for i in range(1, len(x)):
    c1 = lambda x, h: 2 - h*(x*x + 1)
    c2 = lambda x, h: -4
    c3 = lambda x, h: 2 + h*(x*x + 1)

    print(f"c{i - 1} = {c1(x[i], h)}, c{i} = {c2(x[i], h)}, c{i + 1} = {c3(x[i], h)}")
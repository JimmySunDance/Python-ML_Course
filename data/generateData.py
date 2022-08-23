import random as rnd

def straight(x, m, c):
    return m*x+c


data = []

for i in range(70):
    rnd_x = rnd.randrange(0, 100)
    rnd_y = (rnd_x*1.3 + 5)+(rnd.randrange(75))
    data.append(f"{rnd_x},{rnd_y}\n")

with open('data/data1.csv', 'w') as f:
    for l in data:
        f.writelines(l)

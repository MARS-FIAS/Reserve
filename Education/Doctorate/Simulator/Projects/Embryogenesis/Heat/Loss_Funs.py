import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_t = np.power(2, 3) # 7

t = _t - 0 # Variable!

n = t/2
g = t - n

exes = [2/np.power(2, _) for _ in range(0, 4)]
for ex in exes:
    z = np.zeros(t+1)
    print('@', ex, '\nn g')
    for n in range(t+1):
        g = t - n
        x = np.power(n+g, ex)/np.power(_t, ex)
        y = np.power(np.abs(n-g), ex)/np.power(_t, ex)
        z[n] = x-y
        print(n, g)
    plt.plot(z, label = ex)
    print('@', ex, '\n')
z = np.zeros(t+1)
for n in range(t+1):
    g = t - n
    x = n+g
    y = 1+np.abs(n-g)
    z[n] = (x/y)/_t
plt.plot(z, label = '?')
plt.legend()
plt.show()

exes = [2/np.power(2, _) for _ in range(0, 4)]
for ex in exes:
    z = np.zeros(t+1)
    # ex = 2
    for n in range(t+1):
        g = t - n
        x = np.power(n+g, ex)/np.power(_t, ex)
        y = np.power(np.abs(n-g), ex)/np.power(_t, ex)
        z[n] = (np.exp(x-y)-1)/(np.exp(1)-1)
    plt.plot(z, label = ex)
    # plt.show()
z = np.zeros(t+1)
for n in range(t+1):
    g = t - n
    x = n+g
    y = 1+np.abs(n-g)
    z[n] = (x/y)/_t
plt.plot(z, label = '?')
plt.legend()
plt.show()

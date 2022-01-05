import numpy as np
import matplotlib.pyplot as plt


#draw function g
def drawFunction(minx,maxx):
  axes = plt.gca()
  axes.set_xlim([minx,maxx])
  axes.set_ylim([-1,maxx])
  curveX = np.linspace(minx,maxx, 200)
  curveY = g(curveX)
  plt.plot(curveX,curveY)
  plt.title("Recherche du minimum")


def g(x):
    return x * x / 5 - 0.4 * x - 0.5

def dg(x):
    return (2 * x - 2) / 5

drawFunction(-5,5)



x = 4
pas = 0.1

# recherche minimum function

for i in range(100):

    grad = dg(x)
    x -= pas   # à modifier

    # créer la fonction g'(x)
    # appliquez la descente du gradient
    if abs(grad) < 1e-9:
        break;

    plt.scatter(x, g(x),  s=50, c='red',  marker='x')
    plt.pause(0.1) # 0.1second between each iteration, usefull for nice animation

plt.show()
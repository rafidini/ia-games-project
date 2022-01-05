import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import math
random.seed(datetime.now())
# SPYDER 
# Tools > Preferences > IPython Console > Graphics > Backend 
# change it from "Inline/En ligne" to "Automatic".
 
cm = plt.cm.RdBu
 

#draw function g
def drawFunction(minn,maxx):
  plt.clf() 
  xx, yy = np.meshgrid(np.linspace(minn,maxx, 200),np.linspace(minn,maxx, 200))
  zz = g(xx,yy)
  axes = plt.gca()
  axes.set_ylim([minn,maxx])
  axes.set_xlim([minn,maxx])
  plt.contourf(xx,yy,zz, 200, cmap=plt.cm.rainbow)
  plt.colorbar() 
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.title("Gradient descent")


def g(x,y):
   v1 = x*x + 0.4*y*y + 0.1
   v2 = 3 * x + 2 * y + 0.2
   return np.maximum(v1,v2)

def dg(x, y):
    v1 = x*x + 0.4*y*y + 0.1
    v2 = 3 * x + 2 * y + 0.2
    if v1 > v2 :
      return 2*x, 0.8*y
    return 3, 2

# la difficulté pour calculer les dérivées partielles de g est la présence de la fonction : max
# il suffit de remarquer que lorsque v1 > v2, g(x,y) = v1 ...

drawFunction(-5,5)
 
color = [ "red", "blue", "yellow", "green", "white"]



for t in range(5): 
  angle = random.uniform(0,628) / 100
  x = 4 * math.cos( angle )
  y = 4 * math.sin( angle )
  
  k = 0.22                
  
  print("=============================================")
  # gradient descent algorithm 
  for i in range(20):

      print(g(x,y))
      dgdx, dgdy = dg(x, y)  
      x -=  k * dgdx
      y -=  k * dgdy
      #k = k/(1+i)   

      if abs(dgdy) < 0.1 and abs(dgdx) < 0.1:
        print('Minimum:', x, y)
        break;  
    
      plt.scatter(x, y,  s=50, c= color[t] ,  marker='x')
  
      plt.pause(0.05) # 0.1second between each iteration, usefull for nice animation
  
plt.show() # wait for windows close event




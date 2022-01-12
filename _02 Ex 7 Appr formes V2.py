import random
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as FNT

######################################################
# (x,y,category)

points= []
N = 30    # number of points per class
K = 3     # number of classes
for i in range(N):
   r = i / N
   for k in range(K):
      t = ( i * 4 / N) + (k * 4) + random.uniform(0,0.2)
      points.append( [ ( r*math.sin(t), r*math.cos(t) ) , k ] )

######################################################
#  outils d'affichage -  NE PAS TOUCHER

def DessineFond():
    iS = ComputeCatPerPixel()
    levels = [-1, 0, 1, 2]
    c1 = ('r', 'g', 'b')
    plt.contourf(XXXX, YYYY, iS, levels, colors = c1)

def DessinePoints():
    c2 = ('darkred','darkgreen','lightblue')
    for point in points:
        coord = point[0]
        cat   = point[1]
        plt.scatter(coord[0], coord[1] ,  s=50, c=c2[cat],  marker='o')

XXXX , YYYY = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))

##############################################################
# On se propose de travailler avec 2 couches de neurones :
# Input => Linear => Relu => Linear => Scores

class Net(nn.Module) :
  def __init__(self):
    super().__init__()
    self.couche1 = torch.nn.Linear(2, 3)
    self.couche2 = torch.nn.Linear(3, 3)

  def forward(self,x):
    x = self.couche1(x)
    x = FNT.relu(x)
    x = self.couche2(x)
    return x

def loss_fct(Y_preds,Y_true, delta = 0):
  err = Y_preds - (Y_true - delta)
  err[err < 0] = 0
  return torch.sum(err)

def plot_hist(hist):
  fig, axs = plt.subplots(1, max(len(hist)-1,2))
  for i, key in enumerate(hist):
    if key!="epochs":
        axs[i].plot(hist["epochs"], hist[key])
        axs[i].set_title(key)
        axs[i].set_xlabel('epochs', fontsize=12)
  plt.show()

def ComputeCatPerPixel():
    s = XXXX.shape
    preds = model(T)
    preds = torch.argmax(preds, axis= 2)
    CCCC = preds.detach().numpy()
    return CCCC

T = torch.zeros((XXXX.shape[0],XXXX.shape[0],2))
T[:,:,0],T[:,:,1] = torch.FloatTensor(XXXX), torch.FloatTensor(YYYY)

model = Net()
iteration = 500
size = int(np.sqrt(len(points)))
hist = {"loss":[],"epochs":[]}
optim = optim.SGD(model.parameters(), lr=0.001)
X = torch.zeros((size,size,2))
X[:,:,0] = torch.FloatTensor([elt[0][0] for elt in points]).reshape(size,size)
X[:,:,1] = torch.FloatTensor([elt[0][1] for elt in points]).reshape(size,size)
Y = torch.Tensor([elt[1] for elt in points]).reshape(size,size)

for i in range(iteration):
  optim.zero_grad() # remet à zéro le calcul du gradient
  preds = model(X) # démarrage de la passe Forward
  loss = loss_fct(preds, Y) # choisit une function de loss de PyTorch
  hist['loss'].append(loss.item())
  hist['epochs'].append(i)
  loss.backward() # effectue la rétropropagation
  optim.step() # algorithme de descente
  if i%20==0:
     #print(f"Iteration : {i}  ErrorTot : {hist['loss'][-1]}")
     DessineFond()
     DessinePoints()
     plt.title(f"Iteration: {i}/{iteration}")
     #plt.pause(2)  # pause avec duree en secondes
     plt.show(block=False)
plot_hist(hist)





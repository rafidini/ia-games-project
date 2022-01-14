# Ex 7 Appr formes V1
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as FNT

######################################################

# (x,y, category)
points = [ [(0.5,0.4),0],
           [(0.8,0.3),0],
		    [(0.3,0.8),0],
		    [(-.4,0.3),1],
		    [(-.3,0.7),1],
		    [(-.7,0.2),1],
		    [(-.4,-.5),1],
		    [(0.7,-.4),2],
		    [(0.5,-.6),2]]
######################################################
#
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
#
#  PROJET

# Nous devons apprendre 3 catégories : 0 1 ou 2 suivant ce couple (x,y)

# Pour chaque échantillon, nous avons comme information [(x,y),cat]

# Construisez une couche Linear pour un échantillon prédit un score pour chaque catégorie

# Le plus fort score est associé à la catégorie retenue
# Pour calculer l'erreur, on connait la bonne catégorie k de l'échantillon de l'échantillon.
# On calcule Err = Sigma_(j=0 à nb_cat) max(0,Sj-Sk)  avec Sj score de la cat j

# Comment interpréter cette formule :
# La grandeur Sj-Sk nous donne l'écart entre le score de la bonne catégorie et le score de la cat j.
# Si j correspond à k, la contribution à l'erreur vaut 0, on ne tient pas compte de la valeur Sj=k dans l'erreur
# Sinon Si cet écart est positif, ce n'est pas bon signe, car cela sous entend que le plus grand
#          score ne correspond pas à la bonne catégorie et donc on obtient un malus.
#          Plus le mauvais score est grand? plus le malus est important.
#       Si cet écart est négatif, cela sous entend que le score de la bonne catégorie est supérieur
#          au score de la catégorie courante. Tout va bien. Mais il ne faut pas que cela influence
#          l'erreur car l'algorithme doit corriger les mauvaises prédictions. Pour cela, max(0,.)
#          permet de ne pas tenir compte de cet écart négatif dans l'erreur.

class Net(nn.Module) :
  def __init__(self):
    super().__init__()
    self.couche1 = nn.Linear(2, 3)

  def forward(self, x):
    x = self.couche1(x)
    #x = FNT.relu(x)
    return x

def loss_fct(Y_preds,Y_true, delta = 2):
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
    T = torch.zeros((s[0],s[1],2))
    T[:,:,0],T[:,:,1] = torch.FloatTensor(XXXX), torch.FloatTensor(YYYY)
    preds = model(T)
    preds = torch.argmax(preds, axis= 2)
    CCCC = preds.detach().numpy()
    return CCCC

model = Net()
iteration = 500
size = len(points)
hist = {"loss":[],"epochs":[]}
optim = optim.SGD(model.parameters(), lr=1e-3)
X = torch.zeros((size,1,2))
X[:,:,0] = torch.FloatTensor([elt[0][0] for elt in points]).reshape(size,1)
X[:,:,1] = torch.FloatTensor([elt[0][1] for elt in points]).reshape(size,1)
Y = torch.FloatTensor([[1 if i==elt[1] else 0 for i in range(3)] for elt in points]).reshape(size,1,3)

for i in range(iteration):
  optim.zero_grad() # remet à zéro le calcul du gradient
  preds = model(X) # démarrage de la passe Forward
  loss = loss_fct(preds, Y)
  hist['loss'].append(loss.item())
  hist['epochs'].append(i)
  loss.backward() # effectue la rétropropagation
  optim.step() # algorithme de descente
  if (i+1)%100==0:
    #print(f"Iteration : {i}  ErrorTot : {hist['loss'][-1]}")
    DessineFond()
    DessinePoints()
    plt.title(f"Iteration: {i+1}/{iteration}")
    #plt.pause(2)  # pause avec duree en secondes
    plt.show(block=False)
plot_hist(hist)
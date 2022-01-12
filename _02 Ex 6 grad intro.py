import torch
import matplotlib.pyplot as plt
import numpy as np

# Comment installer Pytorch : https://pytorch.org/get-started/locally/

####################################################################
#
#  Objectif

# On se propose comme dans l'ex vu en cours de faire apprendre
# à une fonction le comportement de l'opérateur booléen != (différent)
# => 1 si différent   => 0 si égal

# L'apprentissage doit s'effectuer sur le set d'échantillons suivant :
# (4,2)   (6,-3)    (1,1)    (3,3)
# Cela sous-entend que si l'apprentissage réussit, l'évaluation en dehors
# de ces valeurs peut quand même etre erronée.

# La fonction choisie pour l'apprentissage sera : min(a*|xi-yi|,1)
# avec -a- comme unique paramètre d'apprentissage

# la fonction d'erreur sera simplement : |fnt(xi,yi)-verite_i|


####################################################################
#
#  Aspect technique

# Pour forcer les tenseurs à utiliser des nombres flotants,
# nous utilisons la syntaxe suivante :

x = torch.FloatTensor([ 4,  6,  1 , 3 ])
y = torch.FloatTensor([ 2,  -3,  1 , 3 ])
true = torch.FloatTensor([1, 1, 0, 0])

# pour créer notre paramètre d'apprentissage et préciser que pytorch
# devra gérer son calcul de gradient, nous écrivons :

a = torch.FloatTensor([ 0.1 ])
a.requires_grad = True

# Passe FORWARD :
# Essayez de vous passer d'une boucle for.
# Utilisez les tenseurs pour traiter tous les échantillons en parallèle.
# Calculez les valeurs en sortie de la fonction.
# Calculez l'erreur totale sur l'ensemble de nos échantillons.
# Les fonctions mathématiques sur les tenseurs s'utilisent ainsi :
# torch.abs(..) / torch.min(..)  / torch.sum(..)  ...

# Passe BACKWARD :
# Lorsque le calcul de la passe Forward est terminé,
# nous devons lancer la passe Backward pour calculer le gradient.
# Le calcul du gradient est déclenché par la syntaxe :

# tenseur_erreur_totale.backward()

# GRADIENT DESCENT :
# Effectuez la méthode de descente du gradient pour modifier la valeur
# du paramètre d'apprentissage a. Etrangement, il faut préciser à Pytorch
# d'arrêter de calculer le gradient de a en utilisant la syntaxe ci-après.
# De plus, il faut réinitialiser le gradient de a à zéro manuellement :

#  with torch.no_grad() :
#     a -= ??? *  a.grad
#     a.grad.zero_()


# A chaque itération, affichez la valeur de a et de l'erreur totale
epochs = 25
pas = 1e-2
hist = {"loss":[],"a":[],"epochs":[]}

def model(x, y, a):
  preds = torch.ones((x.shape[0],2))
  preds[:,0] = a * torch.abs(x-y)
  preds = torch.min(preds, 1).values
  return preds

def error(preds,true):
  err = torch.sum(torch.abs(preds-true))
  return err

def plot_hist(hist):
  fig, axs = plt.subplots(1, len(hist)-1)
  for i, key in enumerate(hist):
    if key!="epochs":
        axs[i].plot(hist["epochs"], hist[key])
        axs[i].set_title(key)
        axs[i].set_xlabel('epochs', fontsize=12)
  plt.show() 

for i in range(epochs):
  hist["epochs"].append(i)
  # Generate Prediction
  preds = model(x, y, a)

  # Get the loss and perform backpropagation
  loss = error(preds, true)
  print('-'* 10)
  print(preds, true)
  
  hist["loss"].append(loss.item())
  loss.backward()

  # Let's update the weights
  with torch.no_grad():
    hist["a"].append(a.item())
    a -= a.grad * pas
    # Set the gradients to zero
    a.grad.zero_()
    print(f"Epoch {i}/{epochs}: Loss: {loss} a: {a[0]}")

plot_hist(hist)

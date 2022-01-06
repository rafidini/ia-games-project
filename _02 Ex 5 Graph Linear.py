"""
Exercice 5.

Ouvrez le fichier « Ex Graph Linear.py ». Il reprend le code du tracé d'une couche Linear
avec un seul neurone comme vu précédemment. 

Etape 1 : Retirez la boucle for et utilisez le mécanisme de parallélisation vu ci-dessus. 

Etape 2 : Améliorez le réseau : la première couche doit contenir 3 neurones et la deuxième
    1 neurone unique. La fonction d'activation en sortie de la première couche sera la 
    fonction ReLU. Conserver le mécanisme de parallélisation et tracez le graphique associé. 

Etape + : Quelle est la dimension du tenseur en entrée de la deuxième couche ?

Pour extraire les données d'un tenseur vers un tableau numpy, il faut d'abord le dissocier
du graph de calcul en utilisant la fonction detach(). Vous pouvez ensuite utiliser la
fonction .numpy() pour convertir ce tenseur en tableau numpy.   Vous allez obtenir des
tracés correspondant à des fonctions linéaires par morceaux. Cela est normal, car la sortie
d'un neurone + ReLU est une fonction linéaire par morceau et la combinaison linéaire de
plusieurs fonctions linéaires par morceaux est une fonction linéaire par morceaux. De
manière pragmatique, le tracé d'un neurone + ReLU crée une seule discontinuité (une cassure)
sur le tracé. Avec trois neurones sur la première couche, nous pouvons donc avoir un
maximum de trois cassures sur la courbe finale, ce que nous constatons sur nos tracés... 
"""
# Package necessaires
import torch, numpy, matplotlib.pyplot as plt

# layer = torch.nn.Linear(1,1)	# creation de la couche Linear
# activ = torch.nn.ReLU()         # fonction d’activation ReLU
# Lx = numpy.linspace(-2,2,50)    # échantillonnage de 50 valeurs dans [-2,2]
# Ly = []

# #eval
# for x in Lx:
#   input = torch.FloatTensor([x])	# création d’un tenseur de taille 1
#   v1 = layer(input)			        # utilisation du neurone
#   v2 = activ(v1)			        # application de la fnt activation ReLU
#   Ly.append(v2.item())		        # on stocke le résultat dans la liste

# # tracé
# plt.plot(Lx,Ly,'.') 	# dessine un ensemble de points
# plt.axis('equal') 		# repère orthonormé
# plt.show() 			    # ouvre la fenêtre d'affichage



# Exercice 5.

# # Etape 1
# # Input & Parameters
# N = 50
# Lx = numpy.linspace(-2, 2, N)

# # Modeling
# layer = torch.nn.Linear(1, 1)
# activation = torch.nn.ReLU()

# # Apply modeling
# in_layer = torch.FloatTensor(Lx).reshape(50, 1)
# v1 = layer(in_layer)
# v2 = activation(v1)
# Ly = v2.detach().numpy()
# print('input shape:', Lx.shape)
# print('output shape:', Ly.shape)

# # Graph
# plt.plot(Lx,Ly,'.')
# plt.axis('equal')
# plt.show() 


# Etape 2 : Améliorez le réseau : la première couche doit contenir 3 neurones et la deuxième
#    1 neurone unique. La fonction d'activation en sortie de la première couche sera la 
#    fonction ReLU. Conserver le mécanisme de parallélisation et tracez le graphique associé. 

# Input & Parameters
Lx = numpy.linspace(-2, 2, 50)

# Modeling
layer_1 = torch.nn.Linear(1, 3)
activ_1 = torch.nn.ReLU() 
layer_2 = torch.nn.Linear(3, 1)
activ_2 = torch.nn.ReLU() 

input = torch.FloatTensor(Lx).reshape(50, 1)
v1 = layer_1(input)
v2 = activ_1(v1)
v3 = layer_2(v2)
v4 = activ_2(v3)
Ly = v4.detach().numpy()

print('input_layer_1 shape:', Lx.shape)
print('input_layer_2 shape', v2.detach().numpy().shape)
print('output shape:', Ly.shape)

# Graph
plt.plot(Lx,Ly,'.')
plt.axis('equal')
plt.show()
def original():
    import torch, numpy, matplotlib.pyplot as plt

    layer = torch.nn.Linear(1,1)	# creation de la couche Linear
    activ = torch.nn.ReLU()         # fonction d’activation ReLU
    Lx = numpy.linspace(-2,2,50)    # échantillonnage de 50 valeurs dans [-2,2]
    Ly = []

    #eval
    for x in Lx:
      input = torch.FloatTensor([x])	# création d’un tenseur de taille 1
      v1 = layer(input)			        # utilisation du neurone
      v2 = activ(v1)			        # application de la fnt activation ReLU
      Ly.append(v2.item())		        # on stocke le résultat dans la liste

    # tracé
    plt.plot(Lx,Ly,'.') 	# dessine un ensemble de points
    plt.axis('equal') 		# repère orthonormé
    plt.show() 			    # ouvre la fenêtre d'affichage

# Exercice 5.
def new():
    import torch, numpy, matplotlib.pyplot as plt

    # Input & Parameters
    N = 50
    Lx = numpy.linspace(-2, 2, N)
    in_shape = 1
    out_shape = 3

    # Modeling
    layer = torch.nn.Linear(in_shape, out_shape)
    activation = torch.nn.ReLU()
    in_layer = torch.FloatTensor(Lx).reshape(N, 1)

    # Apply modeling
    v1 = layer(in_layer)
    v2 = activation(v1)
    Ly = v2.detach().numpy()
    print('input shape:', Lx.shape)
    print('output shape:', Ly.shape)

    # Graph
    # plt.plot(Lx, Ly, '.')
    # plt.axis('equal')
    # plt.show()

# # Input & Parameters
# Lx = numpy.linspace(-2, 2, 50)

# # Modeling
# layer_1 = torch.nn.Linear(1, 3)
# activ_1 = torch.nn.ReLU() 
# layer_2 = torch.nn.Linear(3, 1)
# activ_2 = torch.nn.ReLU() 

# input = torch.FloatTensor(Lx).reshape(50, 1)
# v1 = layer_1(input)
# v2 = activ_1(v1)
# v3 = layer_2(v2)
# v4 = activ_2(v3)
# Ly = v4.detach().numpy()

# print('input_layer_1 shape:', Lx.shape)
# print('input_layer_2 shape', v2.detach().numpy().shape)
# print('output shape:', Ly.shape)

# # Graph
# plt.plot(Lx,Ly,'.')
# plt.axis('equal')
# plt.show() 

if __name__ == "__main__":
  new()
  #original()


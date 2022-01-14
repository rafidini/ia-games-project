import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import datasets, transforms

####  1 couche Linear
#	Qu1 : quel est le % de bonnes prédictions obtenu au lancement du programme , pourquoi ?
#       L'initialisation alétoire des paramètres du réseau permet de bien prédire certaines données (88.97% de réussite)

#	Qu2 : quel est le % de bonnes prédictions obtenu avec 1 couche Linear ?
#       Le % de bonnes prédictions vaut 90.36% en fin d'entrainement.

#	Qu3 : pourquoi le test_loader n’est pas découpé en batch ?
#       Le test_loader est utilisé uniquement pour évaluer les performances du réseau après l'entrainement. 
#       Le découpage en batch est utile uniquement pour faciliter la mise à jour du gradient qui à lieu uniquement lors de l'entrainement. 

# Qu4 : pourquoi la couche Linear comporte-t-elle 784 entrées ?
#       C'est la taille de chaque échantillon d'entrée.

# Qu5 : pourquoi la couche Linear comporte-t-elle 10 sorties ?
#       C'est le nombre de classes (chiffres de 0 à 9) possibles en sortie. 

####  2 couches Linear
#   Qu6 : quelles sont les tailles des deux couches Linear ?
#         La couche 1 est de taille (784,128) et la couche 2 de taille (128,10).

# 	Qu7 : quel est l’ordre de grandeur du nombre de poids utilisés dans ce réseau ?
#         ?? 784 * 128 = 100 352 ~ 1e5 poids

#	Qu8 : quel est le % de bonnes prédictions obtenu avec 2 couches Linear ?
#         Le % de bonnes prédictions vaut 97.60% en fin d'entrainement.

####  3 couches Linear
#   Qu9 : obtient-on un réel gain sur la qualité des prédictions ?
#         Le % de bonnes prédictions vaut 97.35% en fin d'entrainement dont non.

####  Fonction Softmax
#   Qu10 : pourquoi est il inutile de changer le code de la fonction TestOK ?
#          Le nombre de classes est toujourst fixé à 10 quel que soit l'architecture
#          du réseau utilisé.


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # # Q1-5
        # self.FC1 = nn.Linear(784, 10)

        # # Q6-8
        # self.FC1 = nn.Linear(784, 128)
        # self.FC2 = nn.Linear(128, 10)

        # # Q9
        self.FC1 = nn.Linear(784, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC3 = nn.Linear(64, 10)

    def forward(self, x):
        n = x.shape[0]
        x = x.reshape((n, 784))

        x = self.FC1(x)
        x = F.relu(x)

        x = self.FC2(x)
        x = F.relu(x)

        #output = self.FC3(x)
        output = nn.Softmax(dim=1)(x)
        return output


    def Loss(self, Scores, target):
        # Scores : (64, 10)
        # target : (64)

        # Original 
        # nb = Scores.shape[0]
        # TRange = torch.arange(0, nb, dtype=torch.int64)
        # scores_cat_ideale = Scores[TRange,target]
        # scores_cat_ideale = scores_cat_ideale.reshape(nb,1)
        # delta = 1
        # Scores = Scores + delta - scores_cat_ideale
        # x = F.relu(Scores)
        # err = torch.sum(x)

        n_sample, n_classes = Scores.shape
        TargetScores = torch.zeros(n_sample, n_classes)

        for i in range(n_sample):
            TargetScores[i, target[i].item()] = 1
        
        cross_entropy = nn.CrossEntropyLoss()
        err = cross_entropy(Scores, TargetScores)

        return err


    def TestOK(self,Scores,target):
        pred = Scores.argmax(dim=1, keepdim=True)  # get the index of the max
        pred = pred.reshape(target.shape)
        eq   = pred == target                      # True when correct prediction
        nbOK = eq.sum().item()                     # count
        return nbOK

##############################################################################

def TRAIN(args, model, train_loader, optimizer, epoch):

    for batch_it, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        Scores = model.forward(data)
        loss = model.Loss(Scores, target)
        loss.backward()
        optimizer.step()

        if batch_it % 50 == 0:
            print(f'   It: {batch_it:3}/{len(train_loader):3} --- Loss: {loss.item():.6f}')


def TEST(model, test_loader):
    ErrTot   = 0
    nbOK     = 0
    nbImages = 0

    with torch.no_grad():
        for data, target in test_loader:
            Scores  = model.forward(data)
            nbOK   += model.TestOK(Scores,target)
            ErrTot += model.Loss(Scores,target)
            nbImages += data.shape[0]

    pc_success = 100. * nbOK / nbImages
    print(f'\nTest set:   Accuracy: {nbOK}/{nbImages} ({pc_success:.2f}%)\n')

##############################################################################

def main(batch_size):

    moy, dev = 0.1307, 0.3081
    TRS = transforms.Compose([transforms.ToTensor(), transforms.Normalize(moy,dev)])
    TrainSet = datasets.MNIST('./data', train=True,  download=True, transform=TRS)
    TestSet  = datasets.MNIST('./data', train=False, download=True, transform=TRS)

    train_loader = torch.utils.data.DataLoader(TrainSet , batch_size)
    test_loader  = torch.utils.data.DataLoader(TestSet, len(TestSet))

    model = Net()
    optimizer = torch.optim.Adam(model.parameters())

    TEST(model,  test_loader)
    for epoch in range(40):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f'Train Epoch: {epoch:3}')

        TRAIN(batch_size, model,  train_loader, optimizer, epoch)
        TEST(model,  test_loader)


main(batch_size=64)

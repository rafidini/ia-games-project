import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import datasets, transforms
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## Original
        #self.FC1 = nn.Linear(3*32*32, 128)
        #self.FC2 = nn.Linear(128,10)

        ## Net 1
        #self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size = (3,3))
        #self.FC2 = nn.Linear(32*30*30, 10)

        ## Net 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size = (3,3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size = (3,3))
        self.FC3 = nn.Linear(64*28*28, 10)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        ## Original
        #n = x.shape[0]
        #x = x.reshape((n,3*32*32))
        #x = self.FC1(x)
        #x = F.relu(x)
        #x = self.FC2(x)

        ## Net1
        #print(f'Input_shape: {x.shape}')
        #x = self.conv1(x)
        #x = F.relu(x)
        #print(f'conv1_shape: {x.shape}')
        #x = torch.flatten(x, 1)
        #x = self.FC3(x)
        #print(f'Output_shape: {x.shape}')

        ## Net 2
        #print(f'Input_shape: {x.shape}')
        x = self.conv1(x)
        x = F.relu(x)
        #print(f'conv1_shape: {x.shape}')
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        #print(f'conv2_shape: {x.shape}')
        x = self.FC3(x)
        #print(f'Output_shape: {x.shape}')
        return x


    def Loss(self,Scores,target):
        y = F.softmax(Scores,dim=1)
        err = self.criterion(y,target)
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

    TRS = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    TrainSet = datasets.CIFAR10('./data', train=True,  download=True, transform=TRS)
    TestSet  = datasets.CIFAR10('./data', train=False, download=True, transform=TRS)

    train_loader = torch.utils.data.DataLoader(TrainSet , batch_size)
    test_loader  = torch.utils.data.DataLoader(TestSet, len(TestSet))

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    TEST(model,  test_loader)
    for epoch in range(20):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f'Train Epoch: {epoch:3}')

        TRAIN(batch_size, model,  train_loader, optimizer, epoch)
        TEST(model,  test_loader)


main(batch_size = 64)
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler

from deformation import ADef, Tnorm
from perturbation import PGD
from models_cifar10 import LeNet, ResNet, BasicBlock


path_to_resources = 'resources/'
path_to_results = 'results/'

############
# Settings for PGD adversarial training
RATE = 0.01
SMTH = 1 # irrelevant for PGD
ITRS = 40
############

def cifar_loaders(batch_size, shuffle_test=False): 
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10(path_to_resources, train=True, download=False, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            
        ]))
    test = datasets.CIFAR10(path_to_resources, train=False, 
        transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    return train_loader, test_loader


def train_epoch( model, loader, epoch, optimizer, adversarial='clean', show_batch=False, cuda=False ):

    model.train()
    
    results_loss = 0.
    results_error = 0.
    n_batches = len(loader)
    
    for batch, (x,y) in enumerate(loader):
        if cuda:
            x = x.cuda()
            y = y.cuda()
        
        adv_magn = 0. # magnitude of adversarial modification
        if adversarial.lower() == 'adef':
            x, adv_dict = ADef( x, model, max_iter=ITRS, max_norm=3, smooth=SMTH, ind_candidates=1, verbose=False, overshoot=1 )
            adv_magn = adv_dict['norms'].mean().item()
        elif adversarial.lower() == 'pgd':
            x = PGD( x, y, model, epsilon=0.3, iterations = ITRS, stepsize=RATE, random_start=True, clip=True )
            adv_magn = -1 #not supported
        
        output = model(Variable( x ))
        
        loss = nn.CrossEntropyLoss()( output, Variable(y) )
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        error = (output.data.max(1)[1] != y).sum().item() / x.size(0)
        
        results_loss += loss.item()
        results_error += error
        
        if batch % 10 == 0:
            print('Train.\tEpch: %d\tBtch: %d\tErr: %.3f%%\tLss: %.5f\tAdv: %.5f' % (epoch, batch, 100*error, loss, adv_magn) )
        
        if show_batch and (batch == 0):
            fname = 'mnist_' + adversarial
            fname += '_e%03db%04d' % (epoch, batch)
            pickle.dump( x.detach().cpu(), open(path_to_results + fname + '.sav', 'wb') )
    
    return results_loss/n_batches, results_error/n_batches


def eval_epoch( model, loader, epoch, cuda=False ):
    
    model.eval()
    
    results_loss = 0.
    results_error = 0.
    n_batches = len(loader)
    
    for batch, (x,y) in enumerate(loader):
        
        if cuda:
            x = x.cuda()
            y = y.cuda()
        
        output = model(Variable( x ))
        
        loss = nn.CrossEntropyLoss()( output, Variable(y) )

        error = (output.data.max(1)[1] != y).sum().item() / x.size(0)
        
        results_loss += loss.item()
        results_error += error
        
        if batch % 10 == 0:
            print('Eval.\tEpch: %d\tBtch: %d\tErr: %.3f%%\tLss: %.5f' % (epoch, batch, 100*error, loss) )
            
    return results_loss/n_batches, results_error/n_batches


if __name__ == '__main__':
    batch_size = 50
    # ~ n_epochs = 84 # Training on approx 100k batches if batch_size = 50
    n_epochs = 2
    learning_rate = 0.0001
    
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        # default model:
        model_id = 'A'
    if model_id.lower() == 'a':
        model = LeNet()
        #load previously trained model to continue the training
        model.load_state_dict(torch.load(path_to_resources + 'lenet_e2b50_clean_model.pt'))
        model.eval() #must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.
    elif model_id.lower() == 'b':
        model = ResNet(block=BasicBlock, num_blocks=[2,2,2,2]) #use ResNet18 
        model.load_state_dict(torch.load(path_to_resources + 'resnet_e100b50_clean_model.pt'))
        model.eval()
    if len(sys.argv) > 2:
        adversarial = sys.argv[2]
    else:
        adversarial = 'clean'
        # ~ adversarial = 'pgd'
        # ~ adversarial = 'adef'
 
    if not adversarial.lower() == 'clean':
        print('Settings for adversarial training:\tRate=%.2f\tItrs=%d\tSmth=%.2f' % (RATE, ITRS, SMTH))
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('Using GPU.')
    
    model_name = model.name
    model_name = model_name + '_e%db%d' % (n_epochs,batch_size)
    model_name = model_name + '_' + adversarial
    
    print('Training %s' % (model_name))
    
    train_loader, test_loader = cifar_loaders( batch_size )
    
    if use_gpu:
        model = model.cuda()
    
    optimizer = optim.Adam( model.parameters(), lr=learning_rate )
    optimizer.load_state_dict(torch.load(path_to_resources + 'lenet_e2b50_clean_optimizer.pt'))
    
    progress_train_loss = -np.ones( n_epochs )
    progress_train_error = -np.ones( n_epochs )
    progress_test_loss = -np.ones( n_epochs )
    progress_test_error = -np.ones( n_epochs )
    res_dict = {'progress_train_loss':progress_train_loss,
                'progress_train_error':progress_train_error,
                'progress_test_loss':progress_test_loss,
                'progress_test_error':progress_test_error}
    
    time0 = time.time()
    for epoch in range(n_epochs):
        
        tr_loss, tr_error = train_epoch( model, train_loader, epoch, optimizer, adversarial=adversarial, show_batch=False, cuda=use_gpu )
        te_loss, te_error = eval_epoch( model, test_loader, epoch, cuda=use_gpu )
        
        progress_train_loss[ epoch ] = tr_loss
        progress_train_error[ epoch ] = tr_error
        progress_test_loss[ epoch ] = te_loss
        progress_test_error[ epoch ] = te_error
        
        torch.save( model.state_dict(), path_to_resources + model_name + '_model.pt' )
        pickle.dump( res_dict, open( path_to_results + model_name + '_progress.sav', 'wb' ) )
        torch.save(optimizer.state_dict(), path_to_resources + model_name + '_optimizer.pt')
        torch.save(epoch, path_to_resources + model_name + '_epoch.pt')
        
        minutes = (time.time() - time0)/60.
        print( '\n\nEpoch %d finished.\n\tTime:\t\t% 3.2f min' % (epoch, minutes))
        print( '\tEval. error:\t% 3.2f%%\n' % (100*te_error) )
    print(model.eval())
    
    



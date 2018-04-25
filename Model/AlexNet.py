'''
 This work is a modified version from https://github.com/bearpaw/pytorch-classification
'''
import os
import sys
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),               
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def GAP_vector(output, true, return_x=False):

    #print(output)
    #print(output.shape)
    #print(type(output),type(true))

    conf, pred = output.topk(1, 1, True, True)
    pred = pred.view(len(pred))
    conf = conf.view(len(conf))
    #true = true.t()

    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.vali = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #gap = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        #precgap = GAP_vector(outputs.data, targets.data, return_x=False)

        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        #gap.update(precgap,inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        #print('Finish batch:' +str(batch_idx))

    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    gap = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        precgap = GAP_vector(outputs.data, targets.data, return_x=False)

        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        gap.update(precgap,inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

    return (losses.avg, top1.avg, gap.avg)

def main():
    global best_acc
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    epochs = 100
    use_cuda = True
    best_acc = 0
    old_loss = 1000000
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    # Data
    path = './TRAIN_TEST64/'
    
    X_train = np.load(path+'X_train64.npy').reshape(17510,3,64,64)
    print('read x_train success')
    X_test = np.load(path+'X_test64.npy').reshape(11674,3,64,64)
    print('read x_test success')

    y_train = []
    with open(path+"y_train64.txt", "r") as f:
     for line in f:
         y_train.append(int(line.strip()))
    
    y_test = []
    with open(path+"y_test64.txt", "r") as f:
     for line in f:
         y_test.append(int(line.strip()))
    
    nX_train = torch.from_numpy(X_train).type('torch.FloatTensor')
    ny_train = torch.IntTensor(y_train).type('torch.LongTensor')
    nX_test = torch.from_numpy(X_test).type('torch.FloatTensor')
    ny_test = torch.IntTensor(y_test).type('torch.LongTensor')
    
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    '''
    num_classes = 10
    train_batch = 40
    test_batch = 40
    workers = 4
    
    print('start')
    trainset = data.TensorDataset(nX_train, ny_train)
    trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=False, num_workers = workers)

    testset = data.TensorDataset(nX_test, ny_test)
    testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers = workers)
    
    model = AlexNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay= weight_decay)
    
    if use_cuda == True:
        model = model.cuda()
        criterion = criterion.cuda()
    # Train and val
    for epoch in range(start_epoch, epochs):
        
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_gap = test(testloader, model, criterion, epoch, use_cuda)
        print('Epoch: [%d | %d] LR: %f; Train Loss %f; Test Loss %f; Train acc %f; Test acc %f; Test GAP %f' 
              % (epoch + 1, epochs, optimizer.param_groups[0]['lr'],train_loss, test_loss, train_acc, test_acc, test_gap))
        sys.stdout.flush()
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
            torch.save(model.state_dict(), 'model_'+str(epoch+1))
        if train_loss >= old_loss:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 1.0/2
        old_loss = train_loss

    print('Best acc:')
    print(best_acc)

if __name__ == '__main__':
    main()

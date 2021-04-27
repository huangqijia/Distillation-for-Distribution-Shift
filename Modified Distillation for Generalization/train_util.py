"""
Author: Qijia Huang
"""

import time

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(net,trainloader,testloader,shiftloader, epochs, batch_size, lr, reg, check_pt, log_every_n=100):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    train_loss_list = []
    test_loss_list = []
    shift_loss_list = []
    train_acc_list = []
    test_acc_list = []
    shift_acc_list = []

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()
        train_loss_list.append(train_loss/len(trainloader)) 
        train_acc_list.append(correct / total)
        scheduler.step()

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))
        test_loss_list.append(test_loss/num_val_steps)
        test_acc_list.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(net.state_dict(), check_pt)
            
        shift_loss = 0
        correct = 0
        total = 0   
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(shiftloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                shift_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(shiftloader)
        val_acc = correct / total
        print("Shift Loss=%.4f, Shift acc=%.4f" % (shift_loss / (num_val_steps), val_acc))
        shift_loss_list.append(shift_loss/num_val_steps)
        shift_acc_list.append(val_acc)
    return train_loss_list, test_loss_list, shift_loss_list,train_acc_list, test_acc_list, shift_acc_list


    
def test(net,testloader,shiftloader):
   
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()

    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
#             if batch_idx ==0:
#                 print(F.softmax(outputs, dim=1)[:20])
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Testset Loss=%.4f, Testset test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc))
    
    
    shift_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(shiftloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
#             if batch_idx ==0:
#                 print(F.softmax(outputs, dim=1)[:20])
            
            loss = criterion(outputs, targets)
            shift_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(shiftloader)
    val_acc = correct / total
    print("Shiftset Loss=%.4f, Shiftset test accuracy=%.4f" % (shift_loss / (num_val_steps), val_acc))


def loss_fn_kd(outputs, labels, teacher_outputs, T=4,alpha=0.9):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KL_loss = nn.KLDivLoss(reduce=None)(F.log_softmax(outputs, dim=1),F.softmax(teacher_outputs/T, dim=1))
    CE_loss = nn.CrossEntropyLoss()(outputs, labels)
#     print(KL_loss*(alpha), CE_loss*(1.-alpha))
#     KD_loss = KL_loss*(alpha * T * T) + CE_loss*(1.-alpha)
    KD_loss = KL_loss*(alpha * T * T) + CE_loss*(1.-alpha)

    return KD_loss


def trainKD(student,teacher,trainloader,testloader, epochs, batch_size, lr, reg, check_pt, log_every_n=100):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    criterion = loss_fn_kd

    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        
        print('\nEpoch: %d' % epoch)
        student.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = student(inputs)
            
            with torch.no_grad():
                soft_targets = teacher(inputs)
            
            if epoch==0 and batch_idx ==0:
                print(F.softmax(soft_targets, dim=1)[:20])
                print(F.softmax(soft_targets/4, dim=1)[:20])
            
            loss = criterion(outputs, targets, soft_targets)
#             loss = criterion(outputs, targets, soft_targets)
#             F.log_softmax(outputs/T, dim=1),F.softmax(teacher_outputs/T, dim=1)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()

        scheduler.step()

        """
        Start the testing code.
        """
        student.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(student.state_dict(), check_pt)

            
            
            
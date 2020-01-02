from __future__ import print_function, division

import torch
import torch.nn as nn                               # torchun nöral networkü
import torch.optim as optim                         # optimizer için
from torch.optim import lr_scheduler                # scheduler için
import numpy as np
import torchvision                                  # dataloader için
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

# data augmentation yapmamız lazım. veri setimizdeki veriyi zenginleştirmek için
# training data için hem augmentation, hem normalizasyon yapıyoruz.
# validation data için sadece normalizasyon. onun augmente olmasını istemiyoruz.
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation([90, 270]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# verimizin lokasyonunu belirledik.
data_dir = 'domates_data'
# veriyi hazırlayıp loaderı yaratıyoruz.
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}

# veri setimizin boyutu
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# sınıf isimlerini dosya isimlerinden çekiyor.
class_names = image_datasets['train'].classes
# öğrenmenin hangi cihaz üzerinde olacağını belirliyoruz. bizde cpu olacak büyük ihtimal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data augmentation'u görebilmek için birkaç fotoğrafı görmek istiyoruz. bunun için
# imshow metodunu yazıyoruz.
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.0001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])


# model eğitim methodu
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # en iyi modeli kaydetmek için;
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    trainAccArray = []
    trainLossArray = []
    valAccArray = []
    valLossArray = []

    # her epoch (iterasyon) dönüşünde
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # her epoch'un kendi içinde bir train ve validation fazı olacaktır.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # her epochun bir lossu ve val datayı doğru bilme sayısı var.
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # veriler ve başlıkları yükleniyor.
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # datalarımızı modele verdik.
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) # criterion ile lossu hesapladık.

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() # eğer traindeysek dloss/dx hesaplatıyoruz
                        optimizer.step() # ve modeli optimize edecek adımı atıyoruz.

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == "train":
                trainAccArray.append(epoch_acc)
                trainLossArray.append(epoch_loss)
            if phase == "val":
                valAccArray.append(epoch_acc)
                valLossArray.append(epoch_loss)

            # deep copy the model
            # validation aşamasında en büyük başarıyı yakalıyoruz ve sonra for dışında kaydediyoruz.
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    plt.figure(2)
    plt.plot(trainAccArray, label='Training Acc')
    plt.plot(valAccArray, label='Validation Acc')
    plt.xlabel('Accuracies')
    plt.ylabel('Epochs')
    plt.suptitle("Şekil 1")
    plt.legend()

    plt.figure(3)
    plt.plot(trainLossArray, label='Training Loss')
    plt.plot(valLossArray, label='Validation Loss')
    plt.suptitle("Şekil 2")
    plt.xlabel('Losses')
    plt.ylabel('Epochs')
    plt.legend()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# modeli alıyoruz ve istediğimiz formata getiriyoruz.
model_ft = models.resnet18(pretrained=True)

#for param in model_ft.parameters():
#    param.requires_grad = False        bunu eklersem fully conected harici bütün layerların
#                                       parametrelerini donduruyoruz ve işlem yarı zamanda bitiyor.
#                                       fakat performansı daha az olur.


num_ftrs = model_ft.fc.in_features #fullyy connectedın giriş değer sayısı lazım.
# bizim 2 tane çıkışımız olacağı için fully connected layer'a ayar çekiyoruz
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

# loss functionu belirliyoruz.
criterion = nn.CrossEntropyLoss()

# Optimizerı belirliyoruz. learning rate ve momentum önemli
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# scheduler'ı her 7 adımda Learning rate'i 10%'una düşürecek şekilde ayarlıyoruz.
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# eğitime hazırladığımız modeli train_model methoduna sokuyoruz.
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, 20)
# sonuçları görmek için;
visualize_model(model_ft)

torch.save(model_ft, "./bestmodel.pth")

plt.ioff()
plt.show()
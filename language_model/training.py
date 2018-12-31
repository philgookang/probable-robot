import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from language_model.Encoder import *

class Train:

    def __init__(self):

        # Device configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Hyper parameters
        num_epochs = 5
        num_classes = 10
        learning_rate = 0.001

        # create CNN Model variable
        model = Encoder(num_classes).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # set model
        self.model = model:


    def get_data(self):

        limit = 1000

        pi = ProductImagesM()
        return pi.getList(limit = limit)


    def begin(self):

        # hyper parameters
        batch_size = 100

        image_list = self.get_data()
        train_loader = torch.utils.data.DataLoader(dataset=image_list,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)

                # Forward pass
                features = self.model(images)

                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

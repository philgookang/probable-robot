from Pretrain import *

# fixed seed
#torch.manual_seed(2)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs          = 10
batch_size          = 4 #64
learning_rate       = 0.001
padding             = 2
stride              = 1
num_workers         = (torch.cuda.device_count() * 4) if torch.cuda.device_count() > 1 else 1

# Data parameters
number_of_images    = 30
limit_loading_count = 1
image_width         = 1500
image_height        = 1800

# ------------------

# create model
model = Pretrain(batch_size = batch_size,
                padding = padding,
                stride = stride,
                num_workers = num_workers,
                number_of_images = number_of_images,
                limit_loading_count = limit_loading_count,
                image_width = image_width,
                image_height = image_height)

#########

# get train loader
train_loader = model.train_loader

# count number of items
total_step = len(train_loader)

########

# check for nulti gpu
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# send model to device
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------------------

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.reshape(-1)
        labels = labels.to(device)

        images = images.cuda()
        labels.cuda()

        # Forward pass
        outputs = model(images)

        # convert to long type
        labels = labels.long()

        # --------------

        # get loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

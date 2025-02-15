import torch
import torch.nn as nn
import torch.nn.functional as F


def val(model, val_data_loader, criterion, device):
    """
    Inputs:
    model (torch.nn.Module): The deep learning model to be trained.
    val_data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to compute the training loss.
    device (torch.device): The device (CPU/GPU) to run the evaluation on.

    Outputs:
    Validation loss
    """
    val_running_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_data_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

    model.train()
    return val_running_loss / len(val_data_loader)


def train(model, data_loader, val_data_loader, criterion, optimizer, epochs, device):
    """
    Inputs:
    model (torch.nn.Module): The deep learning model to be trained.
    data_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    val_data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to compute the training loss.
    optimizer (torch.optim.Optimizer): Optimizer used for updating the model parameters.
    epochs (int): Number of training epochs.
    device (torch.device): The device (CPU/GPU) to run the evaluation on.

    Outputs:
    Tuple of (train_loss_arr, val_loss_arr), an array of the training and validation
    losses at each epoch
    """
    train_loss_arr = []
    val_loss_arr = []

    for _ in range(epochs):
        # TODO: write a training loop
        model.train()
        train_running_loss = 0.0
        for _, (inputs, labels) in enumerate(data_loader, 0):
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss_arr.append(train_running_loss / len(data_loader))
        val_loss_arr.append(val(model, val_data_loader, criterion, device))
        # END TODO

    print("Training finished.")
    return train_loss_arr, val_loss_arr


class ConvNet(nn.Module):
    """
    A Convolutional Neural Net with the following layers
    - conv1: convolution layer with 4 output channels, kernel size of 3, stride of 2, padding of 1 (input is a color image)
    - ReLU nonlinearity
    - conv2: convolution layer with 16 output channels, kernel size of 3, stride of 2, padding of 1
    - ReLU nonlinearity
    - conv3: convolution layer with 32 output channels, kernel size of 3, stride of 2, padding of 1
    - ReLU nonlinearity
    - fc1:    fully connected layer with 1024 output features
    - ReLU nonlinearity
    - fc2:   fully connected layer with 4 output features (the number of classes)
    """

    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()
        # TODO: define the network
        self.conv1 = nn.Conv2d(3, 4, 3, 2, 1)
        self.conv2 = nn.Conv2d(4, 16, 3, 2, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)
        self.fc1 = nn.Linear(8 * 8 * 32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.layers = [
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            lambda x: x.view(x.size(0), -1),
            self.fc1,
            nn.ReLU(),
            self.fc2,
        ]
        # END TODO

    def forward(self, x):
        # TODO: create a convnet forward pass
        for layer in self.layers:
            x = layer(x)
        # END TODO

        return x


class ConvNetMaxPooling(nn.Module):
    """
    Same as ConvNet but with added max pooling of stride = 2 and kernel = 2 after each convolutional block's ReLU
    """

    def __init__(self, num_classes=4):
        super(ConvNetMaxPooling, self).__init__()
        # TODO: define network
        self.conv1 = nn.Conv2d(3, 4, 3, 2, 1)
        self.conv2 = nn.Conv2d(4, 16, 3, 2, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.layers = [
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool,
            self.conv3,
            nn.ReLU(),
            self.pool,
            lambda x: x.view(x.size(0), -1),
            self.fc1,
            nn.ReLU(),
            self.fc2,
        ]
        # END TODO

    def forward(self, x):
        # TODO: create a convnet forward pass
        for layer in self.layers:
            x = layer(x)
        # END TODO
        return x


# Custom Batch Normalization implementation
class BatchNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Initialize a custom Batch Normalization module.

        INPUT:
        - num_features (int): The number of features (channels) to be normalized.
        - eps (float): A small value added to the denominator for numerical stability during normalization.
        - momentum (float): The momentum for updating running statistics.
        """
        super(BatchNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weights = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # We use register buffers so that these tensors move with the model to
        # the correct device
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        """
        Forward pass of the custom BatchNormalization module.

        INPUT:
        - x (Tensor of shape (B, C, H, W)): The input tensor (representing a batch) to be normalized.

        OUTPUT:
        - x (Tensor of shape (B, C, H, W): The output tensor after normalization.

        DIMENSIONS:
            - B (Batch size): The number of samples in the batch.
            - C (Channels): The number of feature channels (e.g., 3 for RGB images).
            - H (Height): The height of each image or feature map.
            - W (Width): The width of each image or feature map.

        These variable names (B, C, H, W) are commonly used conventions in PyTorch

        """
        # Hint: You need to understand broadcasting in Pytorch! https://pytorch.org/docs/stable/notes/broadcasting.html
        # The mean and variance vector should be of shape (1, C, 1, 1)
        if self.training:
            # Compute the mean and variance across the batch, height, and width for each channel.
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

            # Update running statistics. Note: squeeze the extra dimensions to match running_mean/var shape.
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean.view(-1)
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var.view(-1)

            # Normalize the batch using the batch statistics.
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # For inference, reshape running_mean and running_var to (1, C, 1, 1) for broadcasting.
            running_mean = self.running_mean.view(1, self.num_features, 1, 1)
            running_var = self.running_var.view(1, self.num_features, 1, 1)
            # Normalize using the running statistics.
            x_norm = (x - running_mean) / torch.sqrt(running_var + self.eps)

        # Scale and shift the normalized tensor.
        x = self.weights.view(1, self.num_features, 1, 1) * x_norm + self.bias.view(
            1, self.num_features, 1, 1
        )
        return x


class ConvNetBN(nn.Module):
    """
    Same as ConvNetMaxPooling but with BatchNormalization layers after each convolution.
    """

    def __init__(self, num_classes=4):
        super(ConvNetBN, self).__init__()
        # TODO: define network
        self.conv1 = nn.Conv2d(3, 4, 3, 2, 1)
        self.conv2 = nn.Conv2d(4, 16, 3, 2, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.layers = [
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            self.bn2,
            nn.ReLU(),
            self.pool,
            self.conv3,
            self.bn3,
            nn.ReLU(),
            self.pool,
            lambda x: x.view(x.size(0), -1),
            self.fc1,
            nn.ReLU(),
            self.fc2,
        ]
        # END TODO

    def forward(self, x):
        # TODO: create a convnet forward pass
        for layer in self.layers:
            x = layer(x)
        # END TODO
        return x


class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Custom Dropout Layer Initialization.

        Parameters:
        - p (float): The dropout probability, which determines the fraction of
          input elements that will be set to zero during training. It should
          be a value between 0 and 1. By default, it is set to 0.5.

        Returns:
        - None
        """
        super(CustomDropout, self).__init__()
        self.p = p

    def forward(self, x):
        """
        Forward Pass of the Custom Dropout Layer.

        Parameters:
        - x (Tensor): The tensor of weights to which dropout will be applied.

        Returns:
        - x (Tensor): The modified input tensor after applying dropout.
        """
        if self.training:
            # TODO: Implement the training behavior of dropout.
            mask = (torch.rand_like(x) > self.p).float()
            x = x * mask / (1.0 - self.p)
            # END TODO
            pass
        else:
            # TODO: Implement the inference behavior of dropout.
            # END TODO
            pass
        return x


class ConvNetDropout(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNetDropout, self).__init__()
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(3, 4, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 16, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(32)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        # Define the network as a sequence of layers
        self.layers = [
            self.conv1,
            self.bn1,
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout after first ReLU
            self.pool,
            self.conv2,
            self.bn2,
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout after second ReLU
            self.pool,
            self.conv3,
            self.bn3,
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout after third ReLU
            self.pool,
            nn.Flatten(),  # Flatten feature maps into a vector
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout after FC1 ReLU
            self.fc2,
        ]

    def forward(self, x):
        # Forward pass: apply each layer in sequence.
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, interm_channel, out_channel, stride=1):
        """
        Inputs:
        in_channel = number of channels in the input to the first convolutional layer
        interm_channel = number of channels in the output of the first convolutional layer
                       = number of channels in the input to the second convolutional layer
        out_channel = number of channels in the output
        stride = stride for convolution, defaults to 1
        """
        super().__init__()
        # 3 convolutional layers: named conv1, conv2, and conv3
        # 2 batch normalization layers: named bn1, bn2

        # TODO: initialize a residual block with the layers specified above
        # END TODO
        pass

    def forward(self, x):
        # TODO: implement the forward function based on the architecture above
        # END TODO
        pass


class ResNet(nn.Module):
    def __init__(
        self, num_blocks, layer1_channel, layer2_channel, out_channel, num_classes=4
    ):
        """
        Inputs:
        num_blocks = number of blocks in a block layer
        layer1_channel = number of channels in the input to the first block layer
        layer2_channel = number of channels in the output of the first block layer
                       = number of channels in the input to the second blcok layer
        out_channel = number of channels in the output
        num_classes = number of classes in the classification output, defaults to 4
        """
        super(ResNet, self).__init__()
        self.first = nn.Sequential(
            nn.LazyConv2d(layer1_channel, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.LazyLinear(num_classes)
        )
        # initialize two layers called layer1 and layer2 with num_blocks residual blocks each.

        # TODO: (important!) implement the block layer function below before this part
        # END TODO

    def block_layer(self, num_blocks, in_channel, out_channel):
        """
        Inputs:
        num_blocks = number of blocks in the block layer
        in_channel = number of input channels to the entire block layer
        out_channel = number of output channels in the output of the entire block layer
        """
        # make use of the ResidualBlock class you've already implemented
        # again, note interm_channel == out_channel for all residual blocks here

        # TODO: implement the block layer which has num_blocks blocks stacked together
        # END TODO
        pass

    def forward(self, x):
        # TODO: implement the forward function based on the architecture described above
        # END TODO
        pass

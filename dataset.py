import torch
from torchvision import transforms, datasets

def make_datasets(batch_size):
    train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_dataset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    return test_dataset, train_dataset    
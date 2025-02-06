import torch
import argparse
from torch import nn, optim
from torchvision import datasets, transforms, models

def save_checkpoint(save_dir,arch,train_dataset,hidden_units,in_features,model):
    checkpoint = {
        'arch': arch,
        'input_size': in_features,
        'hidden_units': hidden_units,
        'output_size': 102,
        'class_to_idx': train_dataset.class_to_idx,
        'state_dict': model.classifier.state_dict()
    }
    if save_dir.strip() == "":
        torch.save(checkpoint, 'checkpoint.pth')
    else:
        torch.save(checkpoint, save_dir + '/checkpoint.pth')

def train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
    }

    try:
        print(f"[INFO] Start training")
        print(f"[INFO] Base model: {arch}")
        print(f"[INFO] hidden_units: {hidden_units}")
        print(f"[INFO] epochs: {epochs}")
        print(f"[INFO] gpu: {gpu}")

        model = models.get_model(arch, weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier[0].in_features if isinstance(model.classifier, set) or isinstance(model.classifier, list) or isinstance(model.classifier, tuple) else model.classifier.in_features
        if hidden_units > 0:
            model.classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_units, 102),
                                             nn.LogSoftmax(dim=1))
        else:
            model.classifier = nn.Sequential(nn.Linear(in_features, 102), nn.LogSoftmax(dim=1))

        device = "gpu" if gpu else "cpu"
        model.to(device)
        # model training

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']

        for epoch in range(epochs):
            running_loss = 0
            train_batch = 0
            total_train_batches = len(train_loader)
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_batch += 1
                print(f"epoch:{epoch + 1}, train batch: {train_batch}/{total_train_batches}")
            else:
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    model.eval()
                    valid_batch = 0
                    total_valid_batches = len(valid_loader)
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        valid_loss += criterion(log_ps, labels)
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                        valid_batch += 1
                        print(f"epoch:{epoch + 1}, validation batch: {valid_batch}/{total_valid_batches}")

                print("epochs: {}/{}".format(epoch + 1, epochs))
                print("Training loss: {:.3f}".format(running_loss / len(train_loader)))
                print("Validation loss: {:.3f}".format(valid_loss / len(valid_loader)))
                print("Validation Accuracy: {:.3f}%".format(accuracy / len(valid_loader) * 100))
                running_loss = 0
                model.train()
        save_checkpoint(save_dir, arch, image_datasets['train'], hidden_units,in_features, model)
    except KeyError:
        print(f"{arch} is an invalid model name.")
    except Exception as e:
        print(e)
        print("an error occurred while loading the model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train', description='Train the neural network to recognize flower specie.')
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--hidden_units', default=0)
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = float(args.learning_rate)
    hidden_units = int(args.hidden_units)
    epochs = int(args.epochs)
    gpu = args.gpu
    if not torch.cuda.is_available() and gpu:
        print("[INFO] GPU is not available, switching to CPU")
        gpu = False
    train(data_dir, save_dir,arch, learning_rate, hidden_units,epochs, gpu)
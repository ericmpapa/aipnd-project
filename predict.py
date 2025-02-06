import torch
import argparse
from torch import nn
from torchvision import  models
from PIL import Image
import numpy as np
import json

def process_image(image):
    aspect_ratio = image.width / image.height
    width = image.width
    height = image.height
    if min(image.width, image.height) >= 256:
        if image.width > image.height:
            width = int(255 * aspect_ratio)
            height = 255
        else:
            width = 255
            height = 255 // aspect_ratio
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = image.resize((int(width), int(height))).crop((left, top, right, bottom))
    arr = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr_norm = (arr - mean) / std
    return torch.from_numpy(arr_norm.transpose((2,0,1))).type(torch.FloatTensor)


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = models.get_model(checkpoint['arch'], weights="DEFAULT")

    for param in model.parameters():
        param.requires_grad = False

    hidden_units = checkpoint['hidden_units']

    if hidden_units > 0:
        model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units, 102),
                                         nn.LogSoftmax(dim=1))
    else:
        model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], 102), nn.LogSoftmax(dim=1))

    model.classifier.load_state_dict(checkpoint['state_dict'])

    model.idx_to_class = {value: key for key, value in checkpoint['class_to_idx'].items()}
    return model

def predict(image_path, checkpoint, category_names, topk, gpu):
    device = "gpu" if gpu else "cpu"

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    with Image.open(image_path) as im:
        image = process_image(im).to(device)
        model = load_model(checkpoint).to(device)
        ps = torch.exp(model(image.unsqueeze(0)))
        top_p_tensor, top_idx = ps.topk(topk, dim=1)

        top_names = []

        if len(top_idx) > 0:
            for k in range(len(top_idx[0])):
                top_names.append(cat_to_name[f"{top_idx[0][k].item()}"])

        top_p = []

        if len(top_p_tensor) > 0:
            for k in range(len(top_p_tensor[0])):
                top_p.append(top_p_tensor[0][k].item())

        return top_p, top_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='predict', description='Predict the name of a flower based on its image')
    parser.add_argument('image_path',help="Set the image path")
    parser.add_argument('checkpoint', help="Set the checkpoint file path")
    parser.add_argument('--top_k', help="Set the number of most probable classes to return", default=1)
    parser.add_argument('--category_names', help="Set the category to name filepath", default="cat_to_name.json")
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    topk = int(args.top_k)
    category_names = args.category_names
    gpu = args.gpu
    if not torch.cuda.is_available() and gpu:
        print("[INFO] GPU is not available, switching to CPU")
        gpu = False
    top_p, top_names = predict(image_path, checkpoint, category_names, topk, gpu)

    print(f"probabilities:{top_p}, category names:{top_names}")
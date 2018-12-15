from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
from . import utils
import os
import pdb
from copy import deepcopy

def extract_feature_model(pretrained_vgg19):
    new_vgg = deepcopy(pretrained_vgg19)
    classifier_layers = list(list(pretrained_vgg19.children())[1].children())[:5]
    new_vgg.classifier = nn.Sequential(*classifier_layers)
    for p in new_vgg.parameters():
        p.requires_grad = False
    return new_vgg

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.image_feature_model = extract_feature_model(models.vgg19(pretrained=True)).to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for batch_idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, sentence, image = dataset[indices[batch_idx]]
            sentence = sentence.to(self.device)
            image2 = image.unsqueeze(0)
            image3 = image2.to(self.device)

            # Some photos in the dataset are black and white - deal with those
            if image3.shape[1] == 1:
                target_image = torch.zeros(1, 3, 224, 224)
                target_image[:, 0, :, :] = image3
                image3 = target_image
            image_features = self.image_feature_model(image3)
            # self.model.zero_grad()
            #
            # sentence_embeddings, image_embeddings = self.model(sentence, tree, image_features)
            # loss = self.criterion(sentence_embeddings, image_embeddings)
            # loss.backward()
            # self.optimizer.step()
            # total_loss += loss.item()

            # if (batch_idx % 50) == 0:
            #     print("\t - Batch {} -- loss = {}".format(batch_idx, loss.item()))

        self.epoch += 1
        ckpt = os.path.join(self.args.save, "tree-lstm-epoch{}.ckpt".format(self.epoch))
        torch.save(self.model.state_dict(), ckpt)
        print("Checkpoint saved to {}".format(ckpt))

        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                ltree, linput, rtree, rinput, label = dataset[idx]
                target = utils.map_label_to_target(label, dataset.num_classes)
                linput, rinput = linput.to(self.device), rinput.to(self.device)
                target = target.to(self.device)
                output = self.model(ltree, linput, rtree, rinput)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                predictions[idx] = torch.dot(indices, torch.exp(output))
        return total_loss / len(dataset), predictions

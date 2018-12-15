import os
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data
from torchvision import transforms

from . import Constants
from .tree import Tree

class COCODataset(data.Dataset):
    def __init__(self, token_file_path, parent_file_path, image_id_file_path, image_dir_path, vocab):
        super(COCODataset, self).__init__()
        self.vocab = vocab
        self.sentences = self.read_sentences(token_file_path)
        self.trees = self.read_trees(parent_file_path)
        self.image_dir_path = image_dir_path
        self.image_filenames = self.read_image_filenames(image_id_file_path)
        print("image_dir_path = ", image_dir_path)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        sentence = deepcopy(self.sentences[i])
        tree = deepcopy(self.trees[i])
        image_filename = deepcopy(self.image_filenames[i])
        image = deepcopy(self.read_image(image_filename)) # Just added deepcopy, untested
        return tree, sentence, image

    @staticmethod
    def read_image_filenames(filename):
        print("Reading image filenames..")
        with open(filename, 'r') as f:
            image_ids = [line for line in tqdm(f.readlines())]
        return image_ids

    def read_sentences(self, filename):
        print("Reading sentences..")
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_trees(self, filename):
        print("Reading trees..")
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long, device='cpu')

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_image(self, image_name):
        image_name = image_name.strip() + ".jpg"
        img = Image.open(os.path.join(self.image_dir_path, image_name))
        coco_transforms = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        return coco_transforms(img)

if __name__ == '__main__':
    from treelstm import Vocab
    import os
    import glob

    tkn_path = "data/coco/annotations/captions_val2014.toks"
    parent_path = "data/coco/annotations/captions_val2014.parents"
    iid_path = "data/coco/annotations/imageIDs_val2014.txt"
    datadir = "/Users/akhil/git-repos/forked-repos/treelstm.pytorch/data/"
    image_path = os.path.join(datadir, "coco/val2014/")

    annodir = "/Users/akhil/git-repos/forked-repos/treelstm.pytorch/data/coco/annotations"
    tknfiles = glob.glob(annodir + '/*.toks')
    vocab_file = os.path.join(datadir, 'coco/coco.vocab')
    v = Vocab(filename=vocab_file,
              data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    set = COCODataset(tkn_path, parent_path, iid_path, image_path, v)
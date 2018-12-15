from __future__ import division
from __future__ import print_function

import os
import random
import logging
import glob
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

# IMPORT CONSTANTS
from treelstm import Constants
# NEURAL NETWORK MODULES/LAYERS
from treelstm import SimilarityTreeLSTM
# DATA HANDLING CLASSES
from treelstm import Vocab
# DATASET CLASS FOR SICK DATASET
from treelstm.coco_dataset import COCODataset
# METRICS CLASS FOR EVALUATION
from treelstm import Metrics
# UTILITY FUNCTIONS
from treelstm import utils
# TRAIN AND TEST HELPER FUNCTIONS
from treelstm import Trainer
# CONFIG PARSER
from config import parse_args

def get_coco_dataloader(path_to_images, path_to_annotations, mode="training"):
    if mode == "training":
        crop_transform = transforms.randomSizedCrop(224)
    else:
        crop_transform = transforms.CenterCrop(224)

    coco_transforms = transforms.Compose([crop_transform, transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    cap = dset.CocoCaptions(root=path_to_images, annFile=path_to_annotations, transform=coco_transforms)
    print("Loaded {} samples from COCO Captions dataset.".format(len(cap)))
    img, target = cap[0]
    print("Image size = ", img.size())
    print("Captions: ", target)
    return DataLoader(cap, batch_size=1, shuffle=True)


# MAIN BLOCK
def main():
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    anno_dir = os.path.join(args.data, 'annotations')

    # write unique words from all token files
    coco_vocab_file = os.path.join(args.data, 'coco.vocab')
    if not os.path.isfile(coco_vocab_file):
        token_files = glob.glob(anno_dir + '/*.toks')
        utils.build_vocab(token_files, coco_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=coco_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> COCO vocabulary size : %d ' % vocab.size())

    # Load all the data sets
    training_tokens = os.path.join(anno_dir, 'captions_train2014.toks')
    training_parents = os.path.join(anno_dir, 'captions_train2014.parents')
    training_image_ids = os.path.join(anno_dir, 'imageIDs_train2014.txt')
    training_image_dir = os.path.join(args.data, 'train2014/')

    val_tokens = os.path.join(anno_dir, 'captions_val2014.toks')
    val_parents = os.path.join(anno_dir, 'captions_val2014.parents')
    val_image_ids = os.path.join(anno_dir, 'imageIDs_val2014.txt')
    val_image_dir = os.path.join(args.data, 'val2014/')

    print("=" * 80); print("Loading training data"); print("=" * 80)
    train_dataset = COCODataset(training_tokens, training_parents, training_image_ids, training_image_dir, vocab)
    print("Loading validation data"); print("=" * 80)
    val_dataset = COCODataset(val_tokens, val_parents, val_image_ids, val_image_dir, vocab)

    logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    logger.debug('==> Size of dev data     : %d ' % len(val_dataset))

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
        vocab.size(),
        args.input_dim,
        args.mem_dim,
        args.image_features_dim,
        args.sparse,
        args.freeze_embed)
    criterion = nn.MSELoss()

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'coco_embed.pth')
    if os.path.isfile(emb_file):
        print("\nFound pre-existing embedding file, loading that.\n")
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.840B.300d'))
        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    model.emb.weight.data.copy_(emb)

    model.to(device), criterion.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer, device)

    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        logger.info('==> Epoch {}, Train \tLoss: {}'.format(epoch, train_loss))


if __name__ == "__main__":
    main()

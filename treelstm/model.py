import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from . import Constants


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state

class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, image_feature_dim, sparsity, freeze):
        """
        Overall multi-modal model that projects the extracted features from the image to the same vector space
        as the sentence embeddings from our tree-lstm module. Forward pass returns the loss (similarity) between
        an image and a sentence.
        Args:
            vocab_size (int)
            in_dim (int)
            mem_dim (int)
            image_feature_dim (int)
            sparsity (bool)
            freeze (bool)
        """
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.child_sum_tree_lstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.image_projection_layer = nn.Linear(image_feature_dim, mem_dim)

    def forward(self, sentence, parse_tree, image_features):
        """
        Args:
            sentence (torch.tensor): sequence of words represented by their word-ids (obtained from vocab)
            parse_tree (Tree)
            image_features (torch.tensor): extracted features from a pre-trained CNN like VGG19 (usually 1x4096)

        Returns:
            state (torch.tensor): Sentence embedding `y_i = TreeLSTM(x_i)`, usually 1x50
            projected_image (torch.tensor): image embedding `z_i = W * VGG(img_i)`, same dimensions as state
        """
        sentence_embeddings = self.emb(sentence)
        state, hidden = self.child_sum_tree_lstm(parse_tree, sentence_embeddings)
        try:
            projected_image = self.image_projection_layer(image_features)
        except RuntimeError:
            pdb.set_trace()
        assert state.shape == projected_image.shape, "Image and sentence embeddings should be in the same vector space."
        return state, projected_image

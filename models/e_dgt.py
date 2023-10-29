import copy

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F

from preprocess.config import BaseOptions
from transformer_block import TransformerBlock
from graph import Graph


class EDGT(nn.Module):
    def __init__(self, params):
        super(EDGT, self).__init__()
        self.d_model = params.embedding_size
        self.video_linear = nn.Linear(params.vid_feat_size, self.d_model)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.n_trans = Transformer(params)
        e_trans_params = copy.deepcopy(params)
        e_trans_params.embedding_size = e_trans_params.vid_num_regions * e_trans_params.vid_num_regions
        e_trans_params.num_heads = 2
        self.e_trans = Transformer(e_trans_params)

        self.gnn = Graph(self.d_model, self.d_model//2, self.d_model, 2)

        self.satt_pool = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            nn.Tanh(),
            nn.Linear(self.d_model // 2, 1),
            nn.Softmax(dim=-2)
        )

        self.merge_frames = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ELU(inplace=True)
        )

    def encode_video(self, x):
        # Dummy method for video encoding
        return torch.ones(x.size()[0], x.size()[1] * x.size()[2], x.size()[3], self.d_model)

    def build_graph(self, x):
        # Dummy method for graph
        return torch.ones(x.shape[0], x.shape[1], x.shape[1])

    def forward(self, video_o, video_f):
        """
        Forward Pass for the DGT model
        :param video_o: Graph features
        :param video_f: Frame features for the video (bsz, num_clips, num_frames, vid_feat_size)
        :return:
        """
        video_f = self.video_linear(video_f)
        video_f = self.gelu(video_f)
        video_f = self.layer_norm(video_f)  # (bsz, num_clips, num_frames, d_model)

        bsz, num_clips, num_frames, num_regions, feat_dim = video_o.size()
        X = self.encode_video(video_o)  # (bsz, num_clips * num_frames, num_regions, d_model)

        # Node Transformer
        X = self.n_trans(X.reshape(bsz*num_clips*num_regions, num_frames, -1))
        X = X.reshape(bsz*num_clips, num_regions, num_frames, -1)

        h_dim = X.shape[-1]
        X = X.reshape(bsz*num_clips*num_frames, num_regions, h_dim)
        A = self.build_graph(X)  # Build the Adjacency matrix (bsz*num_clips*num_frames, num_regions, num_regions)
        # Edge Transformer
        A = A.view(bsz*num_clips, num_frames, num_regions * num_regions)
        A = self.e_trans(A)
        A = A.view(bsz*num_clips*num_frames, num_regions, num_regions)
        A = F.softmax(A, dim=-1)
        # Apply Graph Convolutions
        X_o = self.gnn(X, A)
        X_o += X

        satt = self.satt_pool(X_o)
        X_o = torch.sum(X_o*satt, dim=-2)

        X_o = X_o.view(bsz, num_clips, num_frames, -1)
        video_rep = self.merge_frames(torch.cat([video_f, X_o], dim=-1))

        # Mean across all the frames in the clip
        video_rep = video_rep.mean(dim=-2)

        return video_rep


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.n_layers = params.num_transformer_layers
        layers = []
        for i in range(self.n_layers):
            layers.append(TransformerBlock(params.embedding_size, params.num_heads, h_dim=1024, num_layers=2,
                                           dropout=0.1))
        self.trans = nn.Sequential(*layers)

    def forward(self, X):
        q_new = self.trans(X)
        return q_new


def get_fake_inputs():
    batch_size = 16
    video_f = torch.ones(batch_size, 8, 20, 2048)
    video_o = torch.ones(batch_size, 8, 20, 32, 2048)
    return video_o, video_f


if __name__ == '__main__':
    import sys
    sys.argv[1:] = ["--input_streams", "transcript"]
    params = BaseOptions().parse()
    model = EDGT(params)
    test_in = get_fake_inputs()
    test_out = model(*test_in)
    print(test_out.size())

import copy

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

from preprocess.config import BaseOptions
from transformer_block import TransformerBlock
from graph import Graph
from cross_attention import CrossAttention


class EDGT(nn.Module):
    def __init__(self, params):
        super(EDGT, self).__init__()
        self.d_model = params.embedding_size
        self.video_linear = nn.Linear(params.vid_feat_size, self.d_model)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.cross_attention = CrossAttention(self.d_model, params.num_heads, h_dim=self.d_model // 2, num_layers=2,
                                              dropout=0.1)

        # Node Transformer
        self.n_trans = Transformer(params)
        # Edge Transformer
        e_trans_params = copy.deepcopy(params)
        e_trans_params.embedding_size = e_trans_params.vid_num_regions * e_trans_params.vid_num_regions
        e_trans_params.num_heads = 2
        self.e_trans = Transformer(e_trans_params)
        # Global Fusion Transformer
        self.gf_trans = Transformer(params)

        self.gnn = Graph(self.d_model, self.d_model // 2, self.d_model, 2)

        self.satt_pool = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Linear(self.d_model // 2, 1),
            nn.Softmax(dim=-2)
        )

        self.merge_frames = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ELU(inplace=True)
        )

        self.multimodal_cross_attention = CrossAttention(self.d_model, 2, self.d_model // 2)

        self.linear_classifier_with_dropout = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.d_model)
        )

        self.output_linear_layer = nn.Linear(self.d_model * 4, 4)

    def encode_video(self, x):
        # Dummy method for video encoding
        return torch.ones(x.size()[0], x.size()[1] * x.size()[2], x.size()[3], self.d_model)

    def get_video_representations(self, video_o, video_f):
        video_f = self.video_linear(video_f)
        video_f = self.gelu(video_f)
        video_f = self.layer_norm(video_f)  # (bsz, num_clips, num_frames, d_model)

        bsz, num_clips, num_frames, num_regions, feat_dim = video_o.size()
        X = self.encode_video(video_o)  # (bsz, num_clips * num_frames, num_regions, d_model)

        # Node Transformer
        X = self.n_trans(X.reshape(bsz * num_clips * num_regions, num_frames, -1))
        X = X.reshape(bsz * num_clips, num_regions, num_frames, -1)

        h_dim = X.shape[-1]
        X = X.reshape(bsz * num_clips * num_frames, num_regions, h_dim)
        A = self.gnn.build_graph(X)  # Build the Adjacency matrix (bsz*num_clips*num_frames, num_regions, num_regions)

        # Edge Transformer
        A = A.view(bsz * num_clips, num_frames, num_regions * num_regions)
        A = self.e_trans(A)
        A = A.view(bsz * num_clips * num_frames, num_regions, num_regions)
        A = F.softmax(A, dim=-1)

        # Apply Graph Convolutions
        X_o = self.gnn(X, A)
        X_o += X

        satt = self.satt_pool(X_o)
        X_o = torch.sum(X_o * satt, dim=-2)

        X_o = X_o.view(bsz, num_clips, num_frames, -1)
        video_rep = self.merge_frames(torch.cat([video_f, X_o], dim=-1))

        # Mean across all the frames in the clip
        video_rep = video_rep.mean(dim=-2)
        return video_rep

    def get_language_embeddings(self, x, lengths):
        attention_mask = torch.zeros(x.size()[0], x.size()[1]).long()
        for i, seq in enumerate(x):
            attention_mask[i, :lengths[i]] = torch.ones(lengths[i])
        with torch.no_grad():
            embed = self.bert_model(x, attention_mask=attention_mask)
        return embed[0]

    def get_language_representations(self, q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, transcript, transcript_l):
        # Calculate embeddings
        embed_q = self.get_language_embeddings(q, q_l)
        embed_a0 = self.get_language_embeddings(a0, a0_l)
        embed_a1 = self.get_language_embeddings(a1, a1_l)
        embed_a2 = self.get_language_embeddings(a2, a2_l)
        embed_a3 = self.get_language_embeddings(a3, a3_l)
        embed_transcript = self.get_language_embeddings(transcript, transcript_l)

        # Combine Question and answers together
        qa0 = torch.cat([embed_q, embed_a0], dim=1)
        qa1 = torch.cat([embed_q, embed_a1], dim=1)
        qa2 = torch.cat([embed_q, embed_a2], dim=1)
        qa3 = torch.cat([embed_q, embed_a3], dim=1)

        # Apply Cross Attention
        tqa_fused0 = self.cross_attention(qa0, embed_transcript, embed_transcript)
        tqa_fused1 = self.cross_attention(qa1, embed_transcript, embed_transcript)
        tqa_fused2 = self.cross_attention(qa2, embed_transcript, embed_transcript)
        tqa_fused3 = self.cross_attention(qa3, embed_transcript, embed_transcript)

        return tqa_fused0, tqa_fused1, tqa_fused2, tqa_fused3

    def forward(self, q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, transcript, transcript_l, video_o, video_f,
                is_cl=True):
        """
        Forward Pass for the DGT model
        :param q:
        :param q_l:
        :param a0:
        :param a0_l:
        :param a1:
        :param a1_l:
        :param a2:
        :param a2_l:
        :param a3:
        :param a3_l:
        :param transcript_l:
        :param transcript:
        :param is_cl: If it is a cross entropy loss
        :param language_embed: Language embeddings for each answer combination
        :param video_o: Graph features
        :param video_f: Frame features for the video (bsz, num_clips, num_frames, vid_feat_size)
        :return:
        """

        video_rep = self.get_video_representations(video_o, video_f)

        # Attend video based on language
        l0, l1, l2, l3 = self.get_language_representations(q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, transcript,
                                                           transcript_l)
        fused_video_rep_a0 = self.multimodal_cross_attention(video_rep, l0, l0)
        fused_video_rep_a1 = self.multimodal_cross_attention(video_rep, l1, l1)
        fused_video_rep_a2 = self.multimodal_cross_attention(video_rep, l2, l2)
        fused_video_rep_a3 = self.multimodal_cross_attention(video_rep, l3, l3)

        # Add to multimodal transformer
        attended_rep_a0 = self.gf_trans(fused_video_rep_a0)
        attended_rep_a1 = self.gf_trans(fused_video_rep_a1)
        attended_rep_a2 = self.gf_trans(fused_video_rep_a2)
        attended_rep_a3 = self.gf_trans(fused_video_rep_a3)

        global_rep_a0 = attended_rep_a0.mean(dim=1)
        global_rep_a1 = attended_rep_a1.mean(dim=1)
        global_rep_a2 = attended_rep_a2.mean(dim=1)
        global_rep_a3 = attended_rep_a3.mean(dim=1)

        if is_cl:
            global_rep_a0 = self.linear_classifier_with_dropout(global_rep_a0)
            global_rep_a1 = self.linear_classifier_with_dropout(global_rep_a1)
            global_rep_a2 = self.linear_classifier_with_dropout(global_rep_a2)
            global_rep_a3 = self.linear_classifier_with_dropout(global_rep_a3)
            return global_rep_a0, global_rep_a1, global_rep_a2, global_rep_a3
        else:
            combined_answers = torch.cat([global_rep_a0, global_rep_a1, global_rep_a2, global_rep_a3], dim=1)
            output = self.output_linear_layer(combined_answers)
            return output


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
    q = torch.ones(batch_size, 25).long()
    q_l = torch.ones(batch_size).fill_(25).long()
    a = torch.ones(batch_size, 4, 20).long()
    a_l = torch.ones(batch_size, 4).fill_(20).long()
    a0, a1, a2, a3 = [a[:, i, :] for i in range(4)]
    a0_l, a1_l, a2_l, a3_l = [a_l[:, i] for i in range(4)]
    transcript = torch.ones(batch_size, 500).long()
    transcript_l = torch.ones(batch_size).fill_(500).long()

    return q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, transcript, transcript_l, video_o, video_f


if __name__ == '__main__':
    # if torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # else:
    #     device = torch.device('cpu')
    #     print("MPS device not found. Going to CPU")
    # # device = 'cpu'
    import sys

    sys.argv[1:] = ["--input_streams", "transcript"]
    params = BaseOptions().parse()
    model = EDGT(params)
    test_in = get_fake_inputs()
    test_out = model(*test_in, is_cl=False)
    print(test_out.size())

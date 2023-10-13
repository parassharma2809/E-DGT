import torch
from torch import nn
from transformers import BertModel
from preprocess.config import BaseOptions
from .cross_attention import CrossAttention


def mean_pool(outputs):
    return outputs.mean(dim=1)


class TranscriptFusion(nn.Module):
    def __init__(self, params):
        super(TranscriptFusion, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.cross_attention = CrossAttention(params.embedding_size, params.num_heads, h_dim=1024, num_layers=2,
                                              dropout=0.1)
        self.linear_classifier = nn.Linear(params.embedding_size * 4, 4)

    def get_embeddings(self, x, lengths):
        attention_mask = torch.zeros(x.size()[0], x.size()[1]).long()
        for i, seq in enumerate(x):
            attention_mask[i, :lengths[i]] = torch.ones(lengths[i])
        with torch.no_grad():
            embed = self.bert_model(x, attention_mask=attention_mask)
        return embed[0]

    def forward(self, q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, transcript, transcript_l, video_feat, vid_l):
        # Calculate embeddings
        embed_q = self.get_embeddings(q, q_l)
        embed_a0 = self.get_embeddings(a0, a0_l)
        embed_a1 = self.get_embeddings(a1, a1_l)
        embed_a2 = self.get_embeddings(a2, a2_l)
        embed_a3 = self.get_embeddings(a3, a3_l)
        embed_transcript = self.get_embeddings(transcript, transcript_l)

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

        # Apply Mean Pooling across fused sequence length
        tqa_pooled0 = mean_pool(tqa_fused0)
        tqa_pooled1 = mean_pool(tqa_fused1)
        tqa_pooled2 = mean_pool(tqa_fused2)
        tqa_pooled3 = mean_pool(tqa_fused3)

        # Add them together as an input to the linear classifier
        answers = torch.cat([tqa_pooled0, tqa_pooled1, tqa_pooled2, tqa_pooled3], dim=1)

        # Linear classifier
        output = self.linear_classifier(answers)
        return output


def get_fake_inputs():
    batch_size = 16
    q = torch.ones(batch_size, 25).long()
    q_l = torch.ones(batch_size).fill_(25).long()
    a = torch.ones(batch_size, 4, 20).long()
    a_l = torch.ones(batch_size, 4).fill_(20).long()
    a0, a1, a2, a3 = [a[:, i, :] for i in range(4)]
    a0_l, a1_l, a2_l, a3_l = [a_l[:, i] for i in range(4)]
    transcript = torch.ones(batch_size, 500).long()
    transcript_l = torch.ones(batch_size).fill_(500).long()
    return q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, transcript, transcript_l, [], []


if __name__ == '__main__':
    import sys
    sys.argv[1:] = ["--input_streams", "transcript"]
    params = BaseOptions().parse()
    model = TranscriptFusion(params)
    test_in = get_fake_inputs()
    test_out = model(*test_in)
    print(test_out.size())

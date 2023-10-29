import torch
from torch import nn
from transformers import BertModel
from preprocess.config import BaseOptions
from .cross_attention import CrossAttention
from .mlp import MLP


def mean_pool(outputs):
    return outputs.mean(dim=1)


class AudioTranscriptFusion(nn.Module):
    def __init__(self, params):
        super(AudioTranscriptFusion, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.cross_attention = CrossAttention(params.embedding_size, params.num_heads, h_dim=params.embedding_size // 2,
                                              num_layers=2, dropout=0.1)
        self.audio_mlp = MLP(params.audio_dim, params.embedding_size, params.audio_dim * 2, num_layers=2)
        self.audio_transcript_cross_attention = CrossAttention(params.embedding_size, params.num_heads,
                                                               h_dim=params.embedding_size // 2, num_layers=2,
                                                               dropout=0.1)
        self.linear_classifier = nn.Linear(params.embedding_size * 4, 4)

    def get_embeddings(self, x, lengths):
        attention_mask = torch.zeros(x.size()[0], x.size()[1]).long()
        for i, seq in enumerate(x):
            attention_mask[i, :lengths[i]] = torch.ones(lengths[i])
        with torch.no_grad():
            embed = self.bert_model(x, attention_mask=attention_mask)
        return embed[0]

    def forward(self, q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, transcript, transcript_l, video_feat, vid_l,
                audio_feat, audio_l):
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

        # Pass audio through MLP
        embed_audio = self.audio_mlp(audio_feat)

        # Apply Cross Attention between audio and fused text
        atqa_fused0 = self.audio_transcript_cross_attention(tqa_fused0, embed_audio, embed_audio)
        atqa_fused1 = self.audio_transcript_cross_attention(tqa_fused1, embed_audio, embed_audio)
        atqa_fused2 = self.audio_transcript_cross_attention(tqa_fused2, embed_audio, embed_audio)
        atqa_fused3 = self.audio_transcript_cross_attention(tqa_fused3, embed_audio, embed_audio)

        # Apply Mean Pooling across fused sequence length
        atqa_pooled0 = mean_pool(atqa_fused0)
        atqa_pooled1 = mean_pool(atqa_fused1)
        atqa_pooled2 = mean_pool(atqa_fused2)
        atqa_pooled3 = mean_pool(atqa_fused3)

        # Add them together as an input to the linear classifier
        answers = torch.cat([atqa_pooled0, atqa_pooled1, atqa_pooled2, atqa_pooled3], dim=1)

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
    audio_feat = torch.ones(batch_size, 2400, 68).float()
    audio_l = torch.ones(batch_size).fill_(2400).long()
    return q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, transcript, transcript_l, [], [], audio_feat, audio_l


if __name__ == '__main__':
    import sys

    sys.argv[1:] = ["--input_streams", "transcript", "audio"]
    params = BaseOptions().parse()
    model = AudioTranscriptFusion(params)
    test_in = get_fake_inputs()
    test_out = model(*test_in)
    print(test_out.size())

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from json_utils import load_json_l_qa
from transformers import BertTokenizer


class SocialIQ2(Dataset):
    def __init__(self, opt, mode="train"):
        self.raw_train = load_json_l_qa(opt.train_path)
        self.raw_test = load_json_l_qa(opt.test_path)
        self.raw_valid = load_json_l_qa(opt.valid_path)
        self.vfeat_load = opt.vfeat
        if self.vfeat_load:
            self.vid_h5 = h5py.File(opt.vid_feat_path, "r", driver=opt.h5driver)
        self.vid_l = opt.max_vid_l
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

        # Tokenizer for BERT embeddings
        # self.word2idx_path = opt.word2idx_path
        # self.idx2word_path = opt.idx2word_path
        # self.vocab_embedding_path = opt.vocab_embedding_path
        self.embedding_dim = opt.embedding_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        # self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
        # self.offset = len(self.word2idx)

        # set entry keys
        # self.text_keys = ["q", "a0", "a1", "a2", "a3", "transcript_text"]
        self.text_keys = ["q", "a0", "a1", "a2", "a3"]
        self.label_key = "answer_idx"
        self.qid_key = "qid"
        self.vid_name_key = "vid_name"

        # build/load vocabulary
        # if not files_exist([self.word2idx_path, self.idx2word_path, self.vocab_embedding_path]):
        #     print("\nNo cache founded.")
        #     self.build_word_vocabulary(word_count_threshold=opt.word_count_threshold)
        # else:
        #     print("\nLoading cache ...")
        #     self.word2idx = load_pickle(self.word2idx_path)
        #     self.idx2word = load_pickle(self.idx2word_path)
        #     self.vocab_embedding = load_pickle(self.vocab_embedding_path)

    def set_mode(self, mode):
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

    def get_cur_dict(self):
        if self.mode == 'train':
            return self.raw_train
        elif self.mode == 'valid':
            return self.raw_valid
        elif self.mode == 'test':
            return self.raw_test

    def __len__(self):
        return len(self.cur_data_dict)

    def __getitem__(self, index):
        items = []
        cur_vid_name = self.cur_data_dict[index][self.vid_name_key]

        # add text keys
        for k in self.text_keys:
            items.append(self.tokenize(self.cur_data_dict[index][k]))

        # add other keys
        if self.mode == 'test':
            items.append(666)  # this value will not be used
        else:
            items.append(int(self.cur_data_dict[index][self.label_key]))
        for k in [self.qid_key]:
            items.append(self.cur_data_dict[index][k])
        items.append(cur_vid_name)

        # add visual feature
        if self.vfeat_load:
            cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][:self.vid_l])
            # cur_vid_feat = nn.functional.normalize(cur_vid_feat, p=2, dim=1)
        else:
            cur_vid_feat = torch.zeros([2, 2])  # dummy placeholder
        items.append(cur_vid_feat)
        return items

    def tokenize(self, line):
        return torch.tensor([self.tokenizer.encode(line)])

    # def numericalize(self, sentence, eos=True):
    #     """convert words to indices"""
    #     sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
    #                         for w in self.line_to_words(sentence, eos=eos)]  # 1 is <unk>, unknown
    #     return sentence_indices

    # def build_word_vocabulary(self, word_count_threshold=0):
    #     """borrowed this implementation from @karpathy's neuraltalk."""
    #     print("Building word vocabulary starts.\n")
    #     all_sentences = []
    #     for k in self.text_keys:
    #         all_sentences.extend([ele[k] for ele in self.raw_train])
    #
    #     word_counts = {}
    #     for sentence in all_sentences:
    #         for w in self.line_to_words(sentence, eos=False, downcase=True):
    #             word_counts[w] = word_counts.get(w, 0) + 1
    #
    #     vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and w not in self.word2idx.keys()]
    #     print("Vocabulary Size %d (<pad> <unk> <eos> excluded) using word_count_threshold %d.\n" %
    #           (len(vocab), word_count_threshold))
    #
    #     # build index and vocabularies
    #     for idx, w in enumerate(vocab):
    #         self.word2idx[w] = idx + self.offset
    #         self.idx2word[idx + self.offset] = w
    #     print("word2idx size: %d, idx2word size: %d.\n" % (len(self.word2idx), len(self.idx2word)))
    #
    #     # Make glove embedding.
    #     print("Loading glove embedding at path : %s. \n" % self.glove_embedding_path)
    #     glove_full = self.load_glove(self.glove_embedding_path)
    #     print("Glove Loaded, building word2idx, idx2word mapping. This may take a while.\n")
    #     glove_matrix = np.zeros([len(self.idx2word), self.embedding_dim])
    #     glove_keys = glove_full.keys()
    #     for i in tqdm(range(len(self.idx2word))):
    #         w = self.idx2word[i]
    #         w_embed = glove_full[w] if w in glove_keys else np.random.randn(self.embedding_dim) * 0.4
    #         glove_matrix[i, :] = w_embed
    #     self.vocab_embedding = glove_matrix
    #     print("Vocab embedding size is :", glove_matrix.shape)
    #
    #     print("Saving cache files ...\n")
    #     save_pickle(self.word2idx, self.word2idx_path)
    #     save_pickle(self.idx2word, self.idx2word_path)
    #     save_pickle(glove_matrix, self.vocab_embedding_path)
    #     print("Building  vocabulary done.\n")


class Batch(object):
    def __init__(self):
        self.__doc__ = "empty initialization"

    @classmethod
    def get_batch(cls, keys=None, values=None):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        assert keys is not None and values is not None
        for k, v in zip(keys, values):
            setattr(batch, k, v)
        return batch


def pad_collate(data, opt):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    def pad_sequences(sequences, k):
        sequences = [torch.LongTensor(s).squeeze() for s in sequences]
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        if k == 'transcript':
            padded_seqs = torch.zeros(len(sequences), opt.max_transcript_l).long()
            lengths = torch.LongTensor([opt.max_transcript_l for _ in sequences])
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq[:end]
        return padded_seqs, lengths

    def pad_video_sequences(sequences):
        """sequences is a list of torch float tensors"""
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        v_dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), opt.max_transcript_l, v_dim).float()
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq
        lengths = torch.LongTensor([opt.max_transcript_l for _ in sequences])
        return padded_seqs, lengths

    # separate source and target sequences
    column_data = list(zip(*data))
    # text_keys = ["q", "a0", "a1", "a2", "a3", "transcript"]
    text_keys = ["q", "a0", "a1", "a2", "a3"]
    label_key = "answer_idx"
    qid_key = "qid"
    vid_name_key = "vid_name"
    vid_feat_key = "vid"
    all_keys = text_keys + [label_key, qid_key, vid_name_key, vid_feat_key]
    all_values = []
    for i, k in enumerate(all_keys):
        if k in text_keys:
            all_values.append(pad_sequences(column_data[i], k))
        elif k == label_key:
            all_values.append(torch.LongTensor(column_data[i]))
        elif k == vid_feat_key:
            all_values.append(pad_video_sequences(column_data[i]))
        else:
            all_values.append(column_data[i])

    batched_data = Batch.get_batch(keys=all_keys, values=all_values)
    return batched_data


def preprocess_inputs(batched_data, max_transcript_l, max_vid_l, device="cuda:0"):
    """clip and move to target device"""
    max_len_dict = {"transcript": max_transcript_l, "vid": max_vid_l}
    # text_keys = ["q", "a0", "a1", "a2", "a3", "transcript"]
    text_keys = ["q", "a0", "a1", "a2", "a3"]
    label_key = "answer_idx"
    qid_key = "qid"
    vid_feat_key = "vid"
    model_in_list = []
    for k in text_keys + [vid_feat_key]:
        v = getattr(batched_data, k)
        if k in max_len_dict:
            ctx, ctx_l = v
            max_l = min(ctx.size(1), max_len_dict[k])
            if ctx.size(1) > max_l:
                ctx_l = ctx_l.clamp(min=1, max=max_l)
                ctx = ctx[:, :max_l]
            model_in_list.extend([ctx.to(device), ctx_l.to(device)])
        else:
            model_in_list.extend([v[0].to(device), v[1].to(device)])
    target_data = getattr(batched_data, label_key)
    target_data = target_data.to(device)
    qid_data = getattr(batched_data, qid_key)
    # print(model_in_list)
    return model_in_list, target_data, qid_data


if __name__ == "__main__":
    import sys
    from config import BaseOptions
    sys.argv[1:] = ["--input_streams", "transcript"]
    opt = BaseOptions().parse()

    dset = SocialIQ2(opt, mode="valid")
    data_loader = DataLoader(dset, batch_size=10, shuffle=False, collate_fn=lambda b: pad_collate(b, opt))

    for batch_idx, batch in enumerate(data_loader):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_transcript_l, opt.max_vid_l, 'cpu')
        print(model_inputs[0][0].shape)
        print(targets.shape)
        print(targets[0].shape)
        break

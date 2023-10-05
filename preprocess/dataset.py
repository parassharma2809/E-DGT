import h5py
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from json_utils import load_json_l_qa, save_json, load_json
from transformers import BertTokenizer
import pysrt
import os

TRANSCRIPT_FOLDER_PATH = '../data/transcript'


class SocialIQ2(Dataset):
    def __init__(self, opt, mode="train"):
        self.raw_train = load_json_l_qa(opt.train_path)
        self.raw_test = load_json_l_qa(opt.test_path)
        self.raw_valid = load_json_l_qa(opt.valid_path)
        self.video_to_transcripts_dict = {}
        self.vfeat_load = opt.vfeat
        if self.vfeat_load:
            self.vid_h5 = h5py.File(opt.vid_feat_path, "r", driver=opt.h5driver)
        self.vid_l = opt.max_vid_l
        self.mode = mode

        # Tokenizer for BERT embeddings
        self.embedding_dim = opt.embedding_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # set entry keys
        self.text_keys = ["q", "a0", "a1", "a2", "a3", "transcript"]
        # self.text_keys = ["q", "a0", "a1", "a2", "a3"]
        self.label_key = "answer_idx"
        self.qid_key = "qid"
        self.vid_name_key = "vid_name"

        # Build transcript video dict if it doesn't exist
        if not os.path.exists('video_to_transcripts.json'):
            self.build_transcript_video_dict()
        else:
            print('Loading transcripts from cache')
            self.video_to_transcripts_dict = load_json('video_to_transcripts.json')

        # Add transcripts to the data
        self.add_transcripts_to_data()

        self.cur_data_dict = self.get_cur_dict()

    def set_mode(self, mode):
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

    def build_transcript_video_dict(self):
        for file in tqdm(os.listdir(TRANSCRIPT_FOLDER_PATH)):
            if file.endswith('.vtt'):
                subs = pysrt.open(f'{TRANSCRIPT_FOLDER_PATH}/{file}')
                sub_text = '\n'.join(
                    [f'{sub.start.minutes}:{sub.start.seconds}-{sub.end.minutes}:{sub.end.seconds}: {sub.text}' for sub
                     in
                     subs])
                self.video_to_transcripts_dict[file.split('.')[0]] = sub_text
        with open('video_to_transcripts.json', 'w') as f:
            save_json(self.video_to_transcripts_dict, f)

    def add_transcript_to_dict(self, data_dict):
        for k, v in data_dict.items():
            if v['vid_name'] in self.video_to_transcripts_dict.keys():
                data_dict[k]['transcript'] = self.video_to_transcripts_dict[v['vid_name']]
            else:
                data_dict[k]['transcript'] = ''

    def add_transcripts_to_data(self):
        self.add_transcript_to_dict(self.raw_train)
        self.add_transcript_to_dict(self.raw_valid)
        self.add_transcript_to_dict(self.raw_test)

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
        return torch.tensor([self.tokenizer.encode(line, truncation=True, max_length=512)])


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
    text_keys = ["q", "a0", "a1", "a2", "a3", "transcript"]
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
    text_keys = ["q", "a0", "a1", "a2", "a3", "transcript"]
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
    return model_in_list, target_data, qid_data


if __name__ == "__main__":
    import sys
    from config import BaseOptions

    sys.argv[1:] = ["--input_streams", "transcript"]
    opt = BaseOptions().parse()

    # We will ignore the test mode for now as we don't have answer index for test split
    dset = SocialIQ2(opt, mode="valid")
    data_loader = DataLoader(dset, batch_size=10, shuffle=False, collate_fn=lambda b: pad_collate(b, opt))

    for batch_idx, batch in enumerate(data_loader):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_transcript_l, opt.max_vid_l, 'cpu')
        print(model_inputs[0][0].shape)
        print(targets.shape)
        print(targets[0].shape)
        break

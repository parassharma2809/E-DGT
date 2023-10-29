import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os
from preprocess.json_utils import save_json, merge_two_dicts
import numpy as np

from models.transcript_fusion import TranscriptFusion
from preprocess.dataset import SocialIQ2, pad_collate, preprocess_inputs
from preprocess.config import TestOptions


def test(opt, dset, model):
    dset.set_mode(opt.mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    qid2preds = {}
    qid2targets = {}
    for valid_idx, batch in tqdm(enumerate(valid_loader)):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_transcript_l, opt.max_vid_l, device=opt.device)
        # outputs = model(*model_inputs)
        # Changed this based on the number of inputs of the model to be used
        outputs = model(*model_inputs[:-2])
        pred_ids = outputs.data.max(1)[1].cpu().numpy().tolist()
        cur_qid2preds = {qid: pred for qid, pred in zip(qids, pred_ids)}
        qid2preds = merge_two_dicts(qid2preds, cur_qid2preds)
        cur_qid2targets = {qid: target for qid, target in zip(qids, targets)}
        qid2targets = merge_two_dicts(qid2targets, cur_qid2targets)
    return qid2preds, qid2targets


def get_acc_from_qid_dicts(qid2preds, qid2targets):
    qids = qid2preds.keys()
    preds = np.asarray([int(qid2preds[ele]) for ele in qids])
    targets = np.asarray([int(qid2targets[ele]) for ele in qids])
    acc = sum(preds == targets) / float(len(preds))
    return acc


if __name__ == "__main__":
    import sys

    sys.argv[1:] = ["--model_dir", "results_2023_03_25_18_06_20"]
    opt = TestOptions().parse()
    dset = SocialIQ2(opt)
    model = TranscriptFusion(opt)
    model.to(opt.device)
    cudnn.benchmark = True
    model_path = os.path.join("results", opt.model_dir, "best_valid.pth")
    model.load_state_dict(torch.load(model_path))

    all_qid2preds, all_qid2targets = test(opt, dset, model)

    if opt.mode == "valid":
        accuracy = get_acc_from_qid_dicts(all_qid2preds, all_qid2targets)
        print("In valid mode, accuracy is %.4f" % accuracy)

    save_path = os.path.join("results", opt.model_dir, "qid2preds_%s.json" % opt.mode)
    save_json(save_path, all_qid2preds)

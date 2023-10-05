import os
import time
import argparse
from json_utils import save_json, load_json


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        # Basic Training/Testing config
        self.parser.add_argument("--debug", action="store_true", help="debug mode, break all loops")
        self.parser.add_argument("--results_dir_base", type=str, default="../results/results")
        self.parser.add_argument("--log_freq", type=int, default=400, help="print, save training info")
        self.parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
        self.parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=5, help="number of epochs to early stop")
        self.parser.add_argument("--bsz", type=int, default=16, help="mini-batch size")
        self.parser.add_argument("--test_bsz", type=int, default=64, help="mini-batch size for testing")
        self.parser.add_argument("--device", type=int, default=0, help="gpu ordinal, -1 indicates cpu")
        self.parser.add_argument("--no_core_driver", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        self.parser.add_argument("--input_streams", type=str, nargs="+", choices=["transcript", "audio", "video"],
                                 help="input streams for the model")

        # model config
        self.parser.add_argument("--num_layers", type=int, default=1, help="number of layers in classifier")
        self.parser.add_argument("--embedding_size", type=int, default=300, help="word embedding dim")
        self.parser.add_argument("--max_transcript_l", type=int, default=300, help="max length for transcripts")
        self.parser.add_argument("--max_vid_l", type=int, default=480, help="max length for video feature")
        self.parser.add_argument("--vocab_size", type=int, default=0, help="vocabulary size")

        # path config
        self.parser.add_argument("--train_path", type=str, default="../data/qa/qa_train_processed.jsonl",
                                 help="train set path")
        self.parser.add_argument("--valid_path", type=str, default="../data/qa/qa_val_processed.jsonl",
                                 help="valid set path")
        self.parser.add_argument("--test_path", type=str, default="../data/qa/qa_test_processed.jsonl",
                                 help="test set path")
        self.parser.add_argument("--vid_feat_size", type=int, default=2048,
                                 help="visual feature dimension")
        self.initialized = True

    def display_save(self, options, results_dir):
        """save config info for future reference, and print"""
        args = vars(options)  # type == dict
        # Display settings
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(results_dir, 'opt.json')
            save_json(option_file_path, args)

    def parse(self):
        """parse cmd line arguments and do some preprocessing"""
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        results_dir = opt.results_dir_base + time.strftime("_%Y_%m_%d_%H_%M_%S")

        if isinstance(self, TestOptions):
            options = load_json(os.path.join("results", opt.model_dir, "opt.json"))
            for arg in options:
                setattr(opt, arg, options[arg])
        else:
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            self.display_save(opt, results_dir)

        opt.input_streams = [] if opt.input_streams is None else opt.input_streams
        opt.vfeat = True if "video" in opt.input_streams else False
        opt.results_dir = results_dir
        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--model_dir", type=str, help="dir contains the model file")
        # We will ignore the test mode for now as we don't have answer index for test split
        self.parser.add_argument("--mode", type=str, default="valid", help="valid/test")


if __name__ == "__main__":
    opt = BaseOptions().parse()

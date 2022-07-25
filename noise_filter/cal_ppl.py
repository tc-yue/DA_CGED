# -*- coding: utf-8 -*-
# @Time    : 2021/08/22 18:24
# @Author  : Tianchiyue


from torch.utils.data import DataLoader
import logging
from transformers import BertForMaskedLM, BertPreTrainedModel, BertTokenizer, \
    BertModel
from transformers.modeling_bert import BertOnlyMLMHead
from collections import OrderedDict
from transformers import AdamW, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset
import argparse
import sys
import torch
import json
import numpy as np
from utils import *
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss
import time
import re
from copy import deepcopy


class BertDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer):
        self.examples = []
        if "pred" in filename:
            self.load_aug(filename)
        else:
            self.load(filename)
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        print(len(self.examples))

    def load_aug(self, filename):
        f = open(filename, encoding="utf8")
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            wrong_span = [item for item in data["wrong_span"] if
                          item != "[unused1]"]
            correct_span = [item for item in data["correct_span"] if
                            item != "[unused1]"]
            left = list(data["left"])
            right = list(data["right"])
            cnt = 0
            for i, token in enumerate(wrong_span):
                sid = f"wrong{idx}_{cnt}"
                new_wrong_span = deepcopy(wrong_span)
                new_wrong_span[i] = "[MASK]"
                sentence = left + new_wrong_span + right
                pos = len(left) + i
                target = token
                cnt += 1
                info = {}
                info["sid"] = sid
                info["sentence"] = sentence
                info["pos"] = pos + 1
                info["target"] = target
                self.examples.append(info)
            for i, token in enumerate(correct_span):
                sid = f"cor{idx}_{cnt}"
                new_wrong_span = deepcopy(correct_span)
                new_wrong_span[i] = "[MASK]"
                sentence = left + new_wrong_span + right
                pos = len(left) + i
                target = token
                cnt += 1
                info = {}
                info["sid"] = sid
                info["sentence"] = sentence
                info["pos"] = pos + 1
                info["target"] = target
                self.examples.append(info)


    def load(self, filename):
        f = open(filename, encoding="utf8")
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            wrong_span = [item for item in data["wrong_span"] if
                          item != "[unused1]"]
            correct_span = [item for item in data["correct_span"] if
                            item != "[unused1]"]
            text = data["text"]
            start, end = -1, -1
            for pos, token in enumerate(text):
                if token == "[MASK]" and start == -1:
                    start = pos
                if token == "[MASK]" and text[pos + 1] != "[MASK]":
                    end = pos + 1
                    break
            cnt = 0
            for i, token in enumerate(wrong_span):
                sid = f"wrong{idx}_{cnt}"
                new_wrong_span = deepcopy(wrong_span)
                new_wrong_span[i] = "[MASK]"
                sentence = text[:start] + new_wrong_span + text[end:]
                pos = start + i
                target = token
                cnt += 1
                info = {}
                info["sid"] = sid
                info["sentence"] = sentence
                info["pos"] = pos + 1
                info["target"] = target
                self.examples.append(info)
            for i, token in enumerate(correct_span):
                sid = f"cor{idx}_{cnt}"
                new_wrong_span = deepcopy(correct_span)
                new_wrong_span[i] = "[MASK]"
                sentence = text[:start] + new_wrong_span + text[end:]
                pos = start + i
                target = token
                cnt += 1
                info = {}
                info["sid"] = sid
                info["sentence"] = sentence
                info["pos"] = pos + 1
                info["target"] = target
                self.examples.append(info)

    def __getitem__(self, index):
        results = convert_examples_to_features(self.examples[index],
                                               self.max_seq_length,
                                               self.tokenizer)
        res = list(torch.tensor(result, dtype=torch.long) for result in results)
        return tuple(res)

    def __len__(self):
        return len(self.examples)


def convert_examples_to_features(example, max_seq_length, tokenizer,
                                 ):
    """label_cls 0-1, label_ids, seq_labels 序列标注"""

    tokens_a = list(example["sentence"])
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    target = tokenizer.convert_tokens_to_ids(example["target"])

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, target


class MLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
        self.loss = CrossEntropyLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            target=None,

    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        batch_size, seq_len, hidden_dim = sequence_output.size()
        # 将mask处的预测概率取出
        x_mask = (input_ids == 103).unsqueeze(-1)
        masked_output = torch.masked_select(sequence_output, x_mask).view(
            batch_size, -1, hidden_dim)
        prediction_scores = self.cls(masked_output)
        prediction_scores = F.softmax(prediction_scores, dim=-1).squeeze(1)
        return prediction_scores


def predict(args):
    set_seed(7)
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.config_fp)
    test_dataset = BertDataset(args.test_file, args.max_seq_length, tokenizer)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False, num_workers=8, pin_memory=True)

    config = BertConfig.from_pretrained(args.config_fp)
    config.num_labels = 2
    model = MLM.from_pretrained(args.config_fp, config=config)
    logging.info("==Init Model==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logging.info(f"\t Start, gpu:{torch.cuda.device_count()}")
    logging.info("\tstart predict")

    pred_res_filename = args.test_file + ".ppl"
    wf = open(pred_res_filename, "w", encoding="utf8")
    idx = 0
    start = time.time()
    sid2prob = {}
    for step, batch in enumerate(test_data_loader):
        if step % 100 == 1:
            logging.info(f"step : \t{step}")
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            prediction_scores = model(
                input_ids=batch[0], attention_mask=batch[1],
                token_type_ids=batch[2])
        target = batch[3].unsqueeze(-1)
        res = prediction_scores.gather(1, target)
        for sample_idx in range(len(batch[0])):
            info = test_dataset.examples[idx]
            idx += 1
            sid = info["sid"].split("_")[0]
            if sid not in sid2prob:
                sid2prob[sid] = []
            sid2prob[sid].append(str(res[sample_idx].item()))
    for sid, prob in sid2prob.items():
        info = {}
        info["sid"] = sid
        info["prob"] = prob
        wf.write(json.dumps(info, ensure_ascii=False) + "\n")
    end = time.time()
    logging.info(f"\ttotal_cost:{end - start}s")
    logging.info(f"\ttotal:{len(test_data_loader)}s")
    wf.close()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file',
                        default='logs/')
    parser.add_argument('--test_file',
                        default="test.txt")
    parser.add_argument('--config_fp',
                        default='test/')
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument('--val_step', default=10, type=int)
    parser.add_argument('--max_seq_length', default=500, type=int)
    parser.add_argument('--jizhi', default=0, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s%(message)s',
                        filename=args.log_file,
                        filemode='w')

    for arg, value in sorted(vars(args).items()):
        logging.info("argument {}:{}".format(arg, value))
    predict(args)


if __name__ == '__main__':
    main(sys.argv)

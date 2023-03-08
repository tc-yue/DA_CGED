# -*- coding: utf-8 -*-
# @Time    : 2021/08/22 18:24
# @Author  : Tianchiyue
# @File    : run_seq_bert.py

from torch.utils.data import DataLoader
import logging
from collections import OrderedDict
from transformers import AdamW, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from model import BertForTokenClassification
from torch.utils.data import Dataset
import argparse
import sys
import torch
import json
import numpy as np
from utils import *


class BertDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer, mode="train"):
        self.examples = []
        with open(filename, encoding="utf8") as f:
            for lineidx, line in enumerate(f):
                data = json.loads(line.strip())
                if "char" in filename:
                    new_wrong = []
                    wrong = data["char_wrong"]
                    for item in wrong:
                        if len(item) >= 3:
                            new_wrong.append([i.strip() for i in item[:3]])
                    data["wrong"] = new_wrong
                else:
                    if mode in ["dev", "test"]:
                        new_wrong = []
                        wrong = data["wrong"]
                        for item in wrong:
                            if len(item) >= 3:
                                new_wrong.append([i.strip() for i in item[:3]])
                        data["wrong"] = new_wrong
                self.examples.append(data)

        self.label_list = {"S": 1, "R": 2, "M": 3, "W": 4, "$": 1}
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        print(len(self.examples))

    def __getitem__(self, index):
        results = convert_examples_to_features(self.examples[index],
                                               self.label_list,
                                               self.max_seq_length,
                                               self.tokenizer,
                                               True)
        res = list(torch.tensor(result, dtype=torch.long) for result in results)
        return tuple(res)

    def __len__(self):
        return len(self.examples)


def convert_examples_to_features(example, label_list, max_seq_length,
                                 tokenizer, return_seq_label=False):
    """label_cls 0-1, label_ids, seq_labels 序列标注"""

    tokens_a = list(example["text"])

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    seq_labels = [0] * len(tokens_a)
    seq_labels = seq_labels[:(max_seq_length - 2)]
    if len(example["wrong"]) > 0:
        label_cls = 1
        label_ids = [0, 0, 0, 0, 0]
        for item in example["wrong"]:
            start_idx, end_idx, token_label = item
            for idx in range(int(start_idx) - 1, int(end_idx)):
                if idx > len(seq_labels) - 1:
                    continue
                seq_labels[idx] = label_list[token_label]
            token_label_idx = label_list[token_label]
            label_ids[token_label_idx] = 1
    else:
        label_cls = 0
        label_ids = [0, 0, 0, 0, 0]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    if return_seq_label:
        seq_labels = [0] + seq_labels + [0]
        seq_labels += padding
        assert len(seq_labels) == max_seq_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if return_seq_label:
        return input_ids, input_mask, segment_ids, label_cls, label_ids, seq_labels
    return input_ids, input_mask, segment_ids, label_cls, label_ids


def train_model(model, device, train_dataloader, test_dataloader,
                optimizer, scheduler, args, epoch, best_score=0):
    losses = []
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        model.train()
        outputs = model(input_ids=batch[0],
                        attention_mask=batch[1],
                        token_type_ids=batch[2],
                        position_ids=None,
                        head_mask=None,
                        seq_labels=batch[5])
        loss = outputs[0]
        if args.n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(recall_model.parameters(), args.max_grad_norm)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            loss = loss / args.gradient_accumulation_steps
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
        if (step + 1) % args.val_step == 0:
            score, test_loss = eval_model(model, test_dataloader, device)
            info = {}
            info["type"] = "train"
            info["step"] = epoch * len(train_dataloader) + step
            info["loss"] = round(np.average(losses), 4)
            logging.info(
                "\tTrain:epoch:{},step:{},loss:{}".format(epoch,
                                                          info["step"],
                                                          info["loss"]))
            if args.jizhi == 1:
                losses = []
                info = {}
                info["type"] = "test"
                info["step"] = epoch * len(train_dataloader) + step
                info["score"] = score
                logging.info(
                    "\tValid: epoch:{},article, step:{},{}".
                             format(epoch, info["step"], score))
                # report_progress(info)
            if float(score) > best_score:
                torch.save(model.state_dict(), args.output_path)
                best_score = float(score)
                logging.info(
                    "\tbest_valid_epoch:{},step:{},{}".format(
                        epoch, info["step"], score))
    return best_score


def eval_model(model, valid_dataloader, device, mode="valid"):
    eval_loss = [0]
    true_res = []
    pred_res = []
    for valid_step, valid_batch in enumerate(valid_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in valid_batch)
        with torch.no_grad():
            seq_labels = batch[5]
            loss, logits = model(input_ids=batch[0], attention_mask=batch[1],
                                 token_type_ids=batch[2],
                                 position_ids=None, head_mask=None,
                                 labels=batch[3], seq_labels=batch[5])
            pred_logits = torch.nn.functional.softmax(logits, dim=-1)
            # (batch_size,seq_len,4)->(batch_size,seq_len)
            pred_batch = torch.argmax(pred_logits, dim=-1)
            attention_mask = batch[1].detach().cpu().numpy()
            for idx in range(len(pred_batch)):
                seq_end = sum(attention_mask[idx])
                pred = pred_batch[idx][1:seq_end].detach().cpu().numpy()
                seq_label = seq_labels[idx][1:seq_end].detach().cpu().numpy()
                true_res.append(seq_label)
                pred_res.append(pred)

    cnt1, cnt2, cnt3 = 0, 0, 0
    pred1, pred2, pred3 = 0, 0, 0
    gold1, gold2, gold3 = 0, 0, 0
    for label, pred in zip(true_res, pred_res):
        if set(label) & {1, 2, 3, 4}:
            gold1 += 1
            if set(pred) & {1, 2, 3, 4}:
                cnt1 += 1
        if set(pred) & {1, 2, 3, 4}:
            pred1 += 1

        gold_error_cnt = len(set(label) & {1, 2, 3, 4})
        pred_error_cnt = len(set(pred) & {1, 2, 3, 4})
        pred_true_cnt = len(set(label) & {1, 2, 3, 4} & set(pred))

        pred2 += pred_error_cnt
        gold2 += gold_error_cnt
        cnt2 += pred_true_cnt

        for idx in range(len(pred)):
            if pred[idx] == label[idx] and pred[idx] in [1, 2, 3, 4]:
                cnt3 += 1
            if pred[idx] in [1, 2, 3, 4]:
                pred3 += 1
            if label[idx] in [1, 2, 3, 4]:
                gold3 += 1

    precision_1 = round(cnt1 / (pred1 + 0.001), 4)
    recall_1 = round(cnt1 / (gold1 + 0.001), 4)
    f1_1 = round(2 * precision_1 * recall_1 / (precision_1 + recall_1 + 0.001),
                 4)

    precision_2 = round(cnt2 / (pred2 + 0.001), 4)
    recall_2 = round(cnt2 / (gold2 + 0.001), 4)
    f1_2 = round(2 * precision_2 * recall_2 / (precision_2 + recall_2 + 0.001),
                 4)

    precision_3 = round(cnt3 / (pred3 + 0.001), 4)
    recall_3 = round(cnt3 / (gold3 + 0.001), 4)
    f1_3 = round(2 * precision_3 * recall_3 / (precision_3 + recall_3 + 0.001),
                 4)

    logging.info(
        "precision_1:{},recall_1:{},f1_1:{}".format(precision_1,
                                                    recall_1,
                                                    f1_1))
    logging.info(
        "precision_2:{},recall_2:{},f1_2:{}".format(precision_2,
                                                    recall_2,
                                                    f1_2))
    logging.info(
        "precision_3:{},recall_3:{},f1_3:{}".format(
            precision_3, recall_3,
            f1_3))
    if mode == "test":
        return pred_res
    else:
        return f1_3, np.mean(eval_loss)


def train_eval(args):
    """ Train the recall_model """
    set_seed(7)
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.config_fp)

    train_dataset = BertDataset(args.train_file, args.max_seq_length,
                                tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    dev_dataset = BertDataset(args.dev_file, args.max_seq_length,
                              tokenizer, mode="dev")
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)

    test_dataset = BertDataset(args.test_file, args.max_seq_length,
                               tokenizer, mode="test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    if "apd" in args.test_file2:
        test_dataset1 = BertDataset(args.test_file2, args.max_seq_length,
                                   tokenizer, mode="test")
        test_dataloader1 = DataLoader(test_dataset1,
                                     batch_size=args.batch_size,
                                     shuffle=False)


    config = BertConfig.from_pretrained(args.config_fp)
    config.num_labels = 2
    config.num_seq_labels = 5
    config.cnn = False
    config.weight_loss = False
    config.use_res = args.use_res
    model = BertForTokenClassification.from_pretrained(args.config_fp,
                                                       config=config)
    if args.do_continue:
        checkpoint = torch.load(args.pretrain_model_path)
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("module"):
                name = k[7:]
            else:
                name = k
            new_checkpoint[name] = v
        model.load_state_dict(new_checkpoint)
        del checkpoint
        del new_checkpoint
    logging.info("==Init Model==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    t_total = int(len(train_dataset) / args.batch_size) * args.epochs
    args.warmup_steps = int(t_total / 10)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    logging.info(f"\tTraining Start, gpu:{torch.cuda.device_count()}")

    model.zero_grad()
    best_score = 0
    if args.train:
        for epoch_id in range(int(args.epochs)):
            logging.info("\tstart epoch:{}".format(epoch_id))
            best_score = train_model(model, device, train_dataloader,
                                     dev_dataloader, optimizer, scheduler, args,
                                     epoch=epoch_id, best_score=best_score)

    if args.test:
        logging.info(f"Test Start")
        checkpoint = torch.load(args.output_path)
        # new_checkpoint = OrderedDict()
        # for k, v in checkpoint.items():
        #     if k.startswith("module"):
        #         name = k[7:]
        #     else:
        #         name = k
        #     new_checkpoint[name] = v
        model.load_state_dict(checkpoint)
        if "apd" in args.test_file2:
            pred_res = eval_model(model, test_dataloader1, device, mode="test")
        pred_res = eval_model(model, test_dataloader, device, mode="test")
        tagid2value = {1: "S", 2: "R", 3: "M", 4: "W"}
        new_res = []
        with open(args.test_file, encoding="utf8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line.strip())

                pred_value = pred_res[idx]
                pos = 0
                flag = False
                while pos < len(pred_value):
                    if pred_value[pos] not in [1, 2, 3, 4]:
                        pos += 1
                        continue
                    if pred_value[pos] in [1, 2, 3, 4]:
                        flag = True
                        start = pos
                        pos += 1
                        if pos >= len(pred_value):
                            break
                        while pred_value[pos] == pred_value[start]:
                            pos += 1
                            if pos >= len(pred_value):
                                break
                        # 标注数据是从1开始数，且闭区间比如【9,10】 包括9，10，对应的是【8：10】
                        new_res.append([data["sid"], str(start + 1), str(pos),
                                        tagid2value[pred_value[start]], "的"])
                if not flag:
                    new_res.append([data["sid"], "correct"])
        with open(args.predict_file, "w", encoding="utf8") as f:
            for line in new_res:
                f.write(",\t".join(line) + "\n")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file',
                        default='logs/')
    parser.add_argument('--output_path',
                        default="logs/")
    parser.add_argument('--train_file',
                        default="train.txt")
    parser.add_argument('--dev_file',
                        default="dev.txt")
    parser.add_argument('--test_file',
                        default="test.txt")
    parser.add_argument('--test_file2',
                        default="test.txt")
    parser.add_argument('--predict_file',
                        default="test.txt")
    parser.add_argument('--pretrain_model_path',
                        default="test.txt")
    parser.add_argument('--config_fp',
                        default='test/')
    parser.add_argument('--version_name',
                        default='baseline')
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
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--train_shuffle", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--use_res", action="store_true", default=False)
    parser.add_argument("--do_continue", action="store_true", default=False)

    args = parser.parse_args()
    args.predict_file = "{}{}.pred".format(args.output_path, args.version_name)
    args.output_path = "{}{}.ckpt".format(args.output_path, args.version_name)
    args.log_file = "{}{}.log".format(args.log_file, args.version_name)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s%(message)s',
                        filename=args.log_file,
                        filemode='w')

    for arg, value in sorted(vars(args).items()):
        logging.info("argument {}:{}".format(arg, value))
    if args.eval:
        eval(args)
    else:
        train_eval(args)


if __name__ == '__main__':
    main(sys.argv)

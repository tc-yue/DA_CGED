from torch.utils.data import DataLoader
import logging
from transformers import BertForMaskedLM, BertPreTrainedModel, BertTokenizer
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
from bert_model import BertModel
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss
import time
import re


class BertDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer, mode="train"):
        self.examples = []
        if mode == "test":
            self.load_test(filename)
        else:
            with open(filename, encoding="utf8") as f:
                for _, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                    except:
                        continue
                    self.examples.append(data)
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        print(len(self.examples))

    def load_test(self, filename):
        f = open(filename, encoding="utf8")
        for line in f:
            if "nlpcc" in filename:
                data = line.strip().split("\t")
                try:
                    if data[1] == "0":
                        correction = data[2]
                    else:
                        correction = data[3]
                except:
                    continue
                start = 5
                if start + 7 > len(correction):
                    continue
                candidates = list(range(start, len(correction) - 7))
                sample_num = min(3, len(candidates))
                for start in random.sample(candidates, sample_num):
                    for n in [3, 4]:
                        correct_span = correction[start:start + n]
                        correct_span = \
                            list(correct_span) + \
                            ["[unused1]" for _ in range(8 - len(correct_span))]
                        wrong_span = ['[unused1]' for _ in range(8)]
                        text = list(correction[:start]) +\
                               ["[MASK]"] * 8 + list(correction[start + n:])
                        info = {}
                        info["wrong_span"] = wrong_span
                        info["correct_span"] = correct_span
                        info["text"] = text
                        info["left"] = correction[:start]
                        info["right"] = correction[start + n:]
                        self.examples.append(info)
                        if len(self.examples) % 10000 == 1:
                            logging.info(len(self.examples))
            else:
                start = 2
                data = json.loads(line.strip())
                correction = data["correction"]
                for start in range(start, len(correction) - 7):
                    for n in [3, 4]:
                        correct_span = correction[start:start + n]
                        correct_span = \
                            list(correct_span) + \
                            ["[unused1]" for _ in range(8 - len(correct_span))]
                        wrong_span = ['[unused1]' for _ in range(8)]

                        text = list(correction[:start]) + \
                               ["[MASK]"] * 8 + list(correction[start + n:])
                        info = {}
                        info["wrong_span"] = wrong_span
                        info["correct_span"] = correct_span
                        info["text"] = text
                        info["left"] = correction[:start]
                        info["right"] = correction[start + n:]
                        self.examples.append(info)
                        if len(self.examples) % 10000 == 1:
                            print(len(self.examples))


    def __getitem__(self, index):
        results = convert_examples_to_features(self.examples[index],
                                               self.max_seq_length,
                                               self.tokenizer)
        res = list(torch.tensor(result, dtype=torch.long) for result in results)
        return tuple(res)

    def __len__(self):
        return len(self.examples)


def convert_examples_to_features(example, max_seq_length, tokenizer,
                                 max_constrained_len=8):
    """label_cls 0-1, label_ids, seq_labels 序列标注"""

    tokens_a = list(example["text"])
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
    target = tokenizer.convert_tokens_to_ids(example["wrong_span"])

    constrained_ids = tokenizer.convert_tokens_to_ids(example["correct_span"])
    constrained_input_mask = [1] * len(constrained_ids)
    constrained_segment_ids = [0] * len(constrained_ids)

    padding = [0] * (max_constrained_len - len(constrained_ids))
    constrained_ids += padding
    constrained_input_mask += padding
    constrained_segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, constrained_ids, \
           constrained_input_mask, constrained_segment_ids, target


def train_model(model, device, train_data_loader,
                optimizer, scheduler, args, epoch, best_score=0):
    losses = []
    for step, batch in enumerate(train_data_loader):
        batch = tuple(t.to(device) for t in batch)
        model.train()
        outputs = model(input_ids=batch[0],
                        attention_mask=batch[1],
                        token_type_ids=batch[2],
                        context_ids=batch[3],
                        context_attention=batch[4],
                        context_token_type=batch[5],
                        target=batch[6])
        loss = outputs[0]
        if args.n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            loss = loss / args.gradient_accumulation_steps
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
        if (step + 1) % args.val_step == 0:
            info = {}
            info["type"] = "train"
            info["step"] = epoch * len(train_data_loader) + step
            info["loss"] = round(np.average(losses), 4)
            logging.info(
                "\tTrain:epoch:{},step:{},loss:{}".format(epoch,
                                                          info["step"],
                                                          info["loss"]))
            torch.save(model.state_dict(), args.output_path)

    return best_score


class ConMLM(BertPreTrainedModel):
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
            context_ids=None,
            context_attention=None,
            context_token_type=None,
            target=None,

    ):
        context_vectors = \
            self.bert(context_ids, context_attention, context_token_type)[0]
        batch_size, seq_len = input_ids.size()
        index = np.argwhere(input_ids.cpu().numpy() == 103)
        index = torch.tensor(index[:, 1].reshape(batch_size, -1))
        context_vector = torch.zeros(batch_size, seq_len, 768).to(input_ids.device)
        index = index.unsqueeze(2).repeat(1, 1, 768).to(input_ids.device)
        context_vector.scatter_add_(1, index, context_vectors)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            context_vector=context_vector
        )
        sequence_output = outputs[0]
        x_mask = (input_ids == 103).unsqueeze(-1)
        batch_size, seq_len, hidden_dim = sequence_output.size()
        masked_output = torch.masked_select(sequence_output, x_mask).view(
            batch_size, -1, hidden_dim)
        prediction_scores = self.cls(masked_output).view(
            batch_size * target.size(1), -1)
        loss = self.loss(prediction_scores, target.view(-1))

        combine_output = outputs[1]
        masked_combine_output = torch.masked_select(combine_output, x_mask)\
            .view(batch_size, -1, hidden_dim)
        masked_context_output = torch.masked_select(context_vector, x_mask)\
            .view(batch_size, -1, hidden_dim)

        # MSE loss
        loss_lambda = 0.5
        context_mean = torch.mean(masked_context_output, dim=1)
        masked_mean = torch.mean(masked_combine_output, dim=1)
        mse_loss = F.mse_loss(masked_mean, context_mean)
        loss += loss_lambda * mse_loss
        prediction_scores = F.softmax(prediction_scores, dim=-1)
        return loss, prediction_scores.view(batch_size, target.size(1), -1)


def predict(model, test_data_loader, device, tokenizer, pred_res_filename,
            examples):
    logging.info("start predict")
    wf = open(pred_res_filename, "w", encoding="utf8")
    idx = 0
    used = set()
    for step, batch in enumerate(test_data_loader):
        if step % 1000 == 1:
            logging.info(f"pred: \t{step}")
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            loss, res = model(input_ids=batch[0],
                              attention_mask=batch[1],
                              token_type_ids=batch[2],
                              context_ids=batch[3],
                              context_attention=batch[4],
                              context_token_type=batch[5],
                              target=batch[6])
        res = res.cpu()
        for sample_idx in range(len(batch[0])):
            ids = []
            for i in torch.argmax(res[sample_idx], dim=-1).numpy().reshape(-1):
                ids.append(i)
            pred_res = "".join(
                [i for i in tokenizer.convert_ids_to_tokens(ids)
                 if i != "[unused1]"])
            info = examples[idx]
            idx += 1
            info["wrong_span"] = pred_res
            wrong_sent = info["left"] + info["wrong_span"] + info["right"]
            if wrong_sent in used:
                continue
            if info["wrong_span"] == "".join(
                    [i for i in info["correct_span"] if i != "[unused1]"]):
                continue
            info["wrong_span"] = re.sub(r'#', "", info["wrong_span"])
            info["wrong_span"] = re.sub(r'\[UNK\]', "", info["wrong_span"])
            info["correct_span"] = "".join(
                [i for i in info["correct_span"] if
                 i != "[unused1]"])
            info.pop("text", None)
            used.add(wrong_sent)
            wf.write(json.dumps(info, ensure_ascii=False) + "\n")
    wf.close()


def train_eval(args):
    """ Train the recall_model """
    set_seed(7)
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.config_fp)
    if args.train:
        train_dataset = BertDataset(args.train_file, args.max_seq_length,
                                    tokenizer)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
    if args.test:
        test_dataset = BertDataset(args.test_file, args.max_seq_length,
                                   tokenizer, mode="test")
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False, num_workers=8,
                                     pin_memory=True)

    config = BertConfig.from_pretrained(args.config_fp)
    config.num_labels = 2
    model = ConMLM.from_pretrained(args.config_fp, config=config)
    logging.info("==Init Model==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.train:
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
        for epoch_id in range(int(args.epochs)):
            logging.info("\tstart epoch:{}".format(epoch_id))
            train_model(model, device, train_dataloader, optimizer,
                        scheduler, args, epoch=epoch_id)
    if args.test:
        logging.info(f"\tTest Start, gpu:{torch.cuda.device_count()}")
        checkpoint = torch.load(args.output_path)
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("module") and args.n_gpu <= 1:
                name = k[7:]
            else:
                name = k
            new_checkpoint[name] = v
        model.load_state_dict(new_checkpoint)
        predict(model, test_dataloader, device, tokenizer,
                args.test_predfilename, test_dataset.examples)


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
    parser.add_argument('--predict_file',
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
    parser.add_argument('--jizhi', default=1, type=int)
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
    args = parser.parse_args()

    mode = "train"
    args.test_predfilename = "{}{}.pred".format(args.output_path,
                                                args.version_name)
    args.output_path = "{}{}.ckpt".format(args.output_path, args.version_name)
    args.log_file = "{}{}_{}.log".format(args.log_file, args.version_name, mode)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s%(message)s',
                        filename=args.log_file,
                        filemode='w')

    for arg, value in sorted(vars(args).items()):
        logging.info("argument {}:{}".format(arg, value))
    train_eval(args)


if __name__ == '__main__':
    main(sys.argv)

import json
import random
import re

MAX_LEN = 8
def process_v1(raw_train_file, data_v1):
    cnt = 0
    wf = open(data_v1, "w")
    with open(raw_train_file) as f:
        for _, line in enumerate(f):
            data = json.loads(line.strip())
            text = data["text"]
            correction = data["correction"]
            for idx in range(2, len(text)-5):
                start = idx
                length = random.choice([3, 4, 5])
                end = idx + length
                # 找到一个窗口对应在正确文本中的位置
                raw_span = text[start:end]
                reg = f'{re.escape(text[start-2:start])}(.*?){re.escape(text[end:end+2])}'
                reg = re.compile(reg)
                if raw_span in correction:
                    continue
                else:
                    result = re.search(reg, correction)
                    if result:
                        correct_span = result.group(1)
                        if not correct_span:
                            continue
                        # 找到该正确span
                        pos = re.search(re.escape(correct_span), correction).span()
                        new_res = list(correction[:pos[0]]) + ["[MASK]" for _ in range(MAX_LEN)] + list(correction[pos[1]:])
                        if len(correct_span) > MAX_LEN:
                            continue
                        if len(correct_span) < len(raw_span):
                            print(cnt)
                        info = {}
                        info["idx"] = cnt
                        cnt += 1
                        info["wrong_span"] = list(raw_span) + ["[unused1]" for _ in range(MAX_LEN - length)]
                        info["correct_span"] = list(correct_span)
                        info["text"] = new_res
                        wf.write(json.dumps(info, ensure_ascii=False)+"\n")
    wf.close()

def process_v2(data_v1, data_v2):
    all_data = []
    with open(data_v1) as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
            except:
                print(idx)
                continue
            all_data.append(data)
    print(len(all_data))

    new_data = []
    miss_data = []
    redundant_data = []
    wrong_order_data = []
    for data in all_data:
        wrong_span = "".join(
            [item for item in data["wrong_span"] if item != "[unused1]"])
        correct_span = "".join(data["correct_span"])
        if data["idx"] == 117161:
            break
        if correct_span == "['还', '有', '，']":
            break
        common = set(list(wrong_span)) & set(list(correct_span))
        if len(common) == len(wrong_span) == len(correct_span):
            wrong_order_data.append(data)
        elif len(wrong_span) == len(correct_span) and \
                len(common) >= len(correct_span) - 2:
            new_data.append(data)
        elif len(wrong_span) > len(correct_span) \
                and len(correct_span) == len(common):
            # 多字
            if wrong_span.startswith(correct_span):
                redundant_data.append(data)
        elif len(correct_span) > len(wrong_span) and len(common) == len(
                wrong_span):
            miss_data.append(data)

    for data in miss_data:
        wrong_span = [item for item in data["wrong_span"] if
                      item != "[unused1]"]
        correct_span = data["correct_span"]
        new_wrong = []
        for correct in correct_span:
            if correct in wrong_span:
                new_wrong.append(correct)
                wrong_span.pop(0)
            else:
                new_wrong.append('[unused1]')
        data["wrong_span"] = new_wrong
    all_data_2 = new_data + miss_data + redundant_data + wrong_order_data
    wf = open(data_v2, "w")
    for data in all_data_2:
        wrong_span = data["wrong_span"]
        correct_span = data["correct_span"]
        wrong_span += ['[unused1]'] * (8 - len(wrong_span))
        correct_span += ['[unused1]'] * (8 - len(correct_span))
        data["wrong_span"] = wrong_span
        data["correct_span"] = correct_span
        wf.write(json.dumps(data, ensure_ascii=False) + "\n")

def process_v3(raw_train_file, data_v2, data_v3):
    cor2wrong = {}
    with open(raw_train_file) as f:
        for line in f:
            data = json.loads(line.strip())
            cor = data["correction"]
            cor2wrong[cor] = data

    all_data = []
    with open(data_v2) as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
            except:
                print(idx)
                continue
            all_data.append(data)
    print(len(all_data))
    cnt = 0

    v3_f = open(data_v3, "w")
    for idx, item in enumerate(all_data):
        wrong = "".join([i for i in item["wrong_span"] if i != '[unused1]'])
        cor = "".join([i for i in item["correct_span"] if i != '[unused1]'])
        start = -1
        end = -1
        for pos, token in enumerate(item["text"]):
            if start == -1 and token == "[MASK]":
                start = pos
                break

        for pos, token in enumerate(item["text"]):
            if token == "[MASK]" and item["text"][pos + 1] != '[MASK]':
                end = pos + 1
                break
        text = item["text"]
        wrong_sentence = "".join(text[:start]) + wrong + "".join(text[end:])
        correct_sentence = "".join(text[:start]) + cor + "".join(text[end:])
        info = cor2wrong[correct_sentence]
        new_wrong = info["text"]
        wrong_chunk = text[start - 1] + wrong + text[end]

        if wrong_chunk not in new_wrong and idx not in [301]:
            cnt += 1
            continue
        v3_f.write(json.dumps(item, ensure_ascii=False) + "\n")

def process_v4(data_v3, data_v4):
    cnt = 0
    r_cnt = 0
    m_cnt = 0
    s_cnt = 0
    w_cnt = 0
    wf = open(data_v4, "w", encoding="utf8")
    with open(data_v3) as f:
        for line in f:
            cnt += 1
            data = json.loads(line.strip())
            wrong_span = [item for item in data['wrong_span'] if
                          item != '[unused1]']
            correct_span = [item for item in data['correct_span'] if
                            item != '[unused1]']
            if len(wrong_span) > len(correct_span):
                r_cnt += 1
                wf.write(line)
                wf.write(line)
            else:
                wf.write(line)
            if len(wrong_span) < len(correct_span):
                m_cnt += 1
            if len(wrong_span) == len(correct_span):
                if len(wrong_span) == len(set(wrong_span) & set(correct_span)):
                    w_cnt += 1
                else:
                    s_cnt += 1
    wf.close()
    print(cnt, r_cnt, m_cnt, w_cnt, s_cnt)

if __name__ == "__main__":
    file_dir = "./data/"
    raw_train_file = "./data/train_processed.txt"
    data_v1 = f"{file_dir}constrained_train_v1.txt"
    data_v2 = f"{file_dir}constrained_train_v2.txt"
    data_v3 = f"{file_dir}constrained_train_v3.txt"
    data_v4 = f"{file_dir}constrained_train_v4.txt"
    process_v1(raw_train_file, data_v1)
    process_v2(data_v1, data_v2)
    process_v3(raw_train_file, data_v2, data_v3)
    process_v4(data_v3, data_v4)

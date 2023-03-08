import json
import re
from copy import deepcopy
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlm_version',
                        default='transformer')
    parser.add_argument('--dir_name',
                        default="./")

    args = parser.parse_args()
    raw_filename = f"{args.dir_name}/data/train_processed.txt"
    correct2raw_info = {}
    raw_lines = []
    with open(raw_filename) as f:
        for line in f:
            data = json.loads(line.strip())
            correct = data["correction"]
            correct2raw_info[correct] = data
            raw_lines.append(data)
    print(len(raw_lines))

    all_info = []
    cnt = 0
    filename =  f"{args.dir_name}/logs/{args.mlm_version}.all"
    with open(filename) as f:
        for line in f:
            cnt += 1
            data = json.loads(line.strip())
            correct = data["correction"]
            gouzao_correct_span = data["correct_span"]
            gouzao_wrong_span = data["wrong_span"]
            raw_info = correct2raw_info[correct]
            raw_wrong = raw_info["text"]

            # 找到generated_span 在正确句子的起始位置
            reg = f'{re.escape(gouzao_correct_span)}'
            reg = re.compile(reg)
            result = re.search(reg, correct)
            if not result:
                continue
            start, end = result.span()

            # 根据左右邻居找到该正确span，对应在原始错误句的位置
            reg = f'{re.escape(correct[start - 2:start])}(.*?){re.escape(correct[end:end + 2])}'
            reg = re.compile(reg)
            result = re.search(reg, raw_wrong)
            if not result:
                continue
            start, end = result.span(1)

            # 如果构造的span 在原始错误中有交集则过滤
            wrong_poses = []
            old_label = deepcopy(raw_info["wrong"])
            for item in old_label:
                left, right, label = item
                left = int(left)
                right = int(right)
                wrong_poses.extend(list(range(left - 1, right)))
            if set(list(range(start, end))) & set(wrong_poses):
                continue

            # 将构造的span 塞入，并且加入label
            new_label = []
            new_wrong = raw_wrong[:start] + gouzao_wrong_span + raw_wrong[end:]
            ends = 0

            for item in data["wrong_no_offset"]:
                left, right, label = item
                left = int(left)
                right = int(right)
                left += start
                right += start
                ends = max(ends, right)
                new_label.append([str(left), str(right), str(label)])

            for item in old_label:
                left, right, label = item
                left = int(left)
                right = int(right)
                offset = len(new_wrong) - len(raw_wrong)
                if right < end:
                    new_label.append(item)
                    continue
                else:
                    left += offset
                    right += offset
                    new_label.append([str(left), str(right), str(label)])

            new_info = {}
            new_info["text"] = new_wrong
            new_info["correction"] = correct
            new_info["wrong"] = new_label
            raw_lines.append(new_info)

    wfilename = filename + ".multi"
    wf = open(wfilename, "w", encoding="utf8")
    for info in raw_lines:
        wf.write(json.dumps(info, ensure_ascii=False) + "\n")
    wf.close()

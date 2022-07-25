import json
import numpy as np
import sys
import argparse


def filter_miss(threshold, id2info, id2ppl_info):
    all_info = []
    for idx, info in id2info.items():
        wrong_id = f"wrong{idx}"
        cor_id = f"cor{idx}"
        wrong_ppl = id2ppl_info.get(wrong_id, 1000)
        cor_ppl = id2ppl_info.get(cor_id, 1)

        info["wrong_ppl"] = str(round(wrong_ppl, 4))
        info["cor_ppl"] = str(round(cor_ppl, 4))
        if 5 * cor_ppl <= wrong_ppl < threshold + 1 \
                and wrong_ppl > threshold \
                and "[" not in info["wrong_span"]:
            wrong_span = info["wrong_span"]
            right = info["right"]
            correct_span = info["correct_span"]

            if wrong_span == correct_span:
                continue
            if len(wrong_span) != len(set(list(wrong_span))):
                continue

            wrong_words = set(list(wrong_span))
            correct_words = set(list(correct_span))
            common = set(wrong_words & correct_words)

            if wrong_span[-1] == right[0]:
                continue
            # m label
            if len(correct_span) > len(wrong_span) and len(common) == len(
                    wrong_words) and len(common) < len(correct_words):
                all_info.append(info)
    return all_info


def filter_redundant(threshold, id2info, id2ppl_info):
    all_info = []
    for idx, info in id2info.items():
        wrong_id = f"wrong{idx}"
        cor_id = f"cor{idx}"
        wrong_ppl = id2ppl_info.get(wrong_id, 1000)
        cor_ppl = id2ppl_info.get(cor_id, 1)

        info["wrong_ppl"] = str(round(wrong_ppl, 4))
        info["cor_ppl"] = str(round(cor_ppl, 4))
        if wrong_ppl >= 5 * cor_ppl \
                and wrong_ppl > threshold \
                and wrong_ppl < threshold+1 \
                and "[" not in info["wrong_span"]:
            wrong_span = info["wrong_span"]
            right = info["right"]
            correct_span = info["correct_span"]

            if wrong_span == correct_span:
                continue
            if len(wrong_span) != len(set(list(wrong_span))):
                continue

            wrong_words = set(list(wrong_span))
            correct_words = set(list(correct_span))
            common = set(wrong_words & correct_words)

            if wrong_span[-1] == right[0]:
                continue
            if len(correct_span) < len(wrong_span) and len(common) < len(
                    wrong_words) and len(common) == len(correct_words):
                    # and len(correct_words) == 4:
                all_info.append(info)
    return all_info


def filter_select(threshold, id2info, id2ppl_info):
    all_info = []
    for idx, info in id2info.items():
        wrong_id = f"wrong{idx}"
        cor_id = f"cor{idx}"
        wrong_ppl = id2ppl_info.get(wrong_id, 1000)
        cor_ppl = id2ppl_info.get(cor_id, 1)
        info["wrong_ppl"] = str(round(wrong_ppl, 4))
        info["cor_ppl"] = str(round(cor_ppl, 4))
        if wrong_ppl >= 5 * cor_ppl \
                and wrong_ppl > threshold \
                and wrong_ppl < threshold + 1 \
                and "[" not in info["wrong_span"]:
            wrong_span = info["wrong_span"]
            right = info["right"]
            correct_span = info["correct_span"]

            if wrong_span == correct_span:
                continue
            if len(wrong_span) != len(set(list(wrong_span))):
                continue

            wrong_words = set(list(wrong_span))
            correct_words = set(list(correct_span))
            common = set(wrong_words & correct_words)

            if wrong_span[-1] == right[0]:
                continue
            if len(correct_span) == len(wrong_span) and len(common) >= len(
                    wrong_words) - 2:
                all_info.append(info)
    return all_info

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file',
                        default='logs/')
    parser.add_argument('--ppl_file',
                        default="test.txt")
    parser.add_argument('--denoise_file',
                        default='test/')
    args = parser.parse_args()
    id2info = {}
    with open(args.pred_file) as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            id2info[idx] = data

    id2ppl_info = {}
    with open(args.ppl_file) as f:
        for line in f:
            data = json.loads(line.strip())
            score = data["prob"]
            new_score = -1 * np.sum([np.log(float(item)) for item in score]) / len(score)
            data["ppl"] = new_score
            id2ppl_info[data["sid"]] = new_score

    print("ok")

    all_info = []
    threshold = 2
    infos = filter_redundant(threshold, id2info, id2ppl_info)
    all_info.extend(infos)
    infos = filter_select(threshold, id2info, id2ppl_info)
    all_info.extend(infos)
    infos = filter_miss(threshold, id2info, id2ppl_info)
    all_info.extend(infos)

    with open(args.denoise_file, "w") as f:
        for line in all_info:
            f.write(json.dumps(line, ensure_ascii=False)+ "\n")

if __name__ == '__main__':
    main(sys.argv)

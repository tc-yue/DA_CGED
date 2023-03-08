import jieba
import json
import time
import argparse
from get_text_edits_on_processed_train_data import get_final_text_edits
from get_text_edits_on_char_level import get_text_edits_on_token


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def filter_not_chinese(text, wrong_list, filter_label='S'):
    # 过滤掉特定类型的错误与非中文的错误
    # not_chinese_num=0
    for i in wrong_list:
        # if i[-1]!=filter_label or is_chinese(text[int(i[0])-1])==False:return False
        if is_chinese(text[int(i[0]) - 1]) == False:
            return False
            # not_chinese_num+=1
    # print('not chinese num',not_chinese_num)
    return True


def filter_noise_in_augment_data(filename):
    raw_data_after_ppl = []
    with open(filename, 'r', encoding='utf-8')as f:
        for line in f.readlines():
            raw_data_after_ppl.append(json.loads(line.strip()))
    data_denoise = []
    not_chinese_num = 0
    for idx, line in enumerate(raw_data_after_ppl):
        if filter_not_chinese(line['text'], line['wrong'], 'S'):
            data_denoise.append(line)
        else:
            not_chinese_num += 1
            # print('noise',line)
            continue
    print('not chinese num', not_chinese_num)
    with open(filename, 'w', encoding='utf-8')as f:
        for line in data_denoise:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def get_s_label_word_level_on_whole_sent_pair(raw_data_file, out_file):
    raw_data_after_ppl = []
    with open(raw_data_file, 'r', encoding='utf-8')as f:
        for line in f.readlines():
            raw_data_after_ppl.append(json.loads(line.strip()))

    data_slabel_on_word_level = []
    for idx, line in enumerate(raw_data_after_ppl):
        # for idx,line in enumerate(raw_data_after_ppl[:200]):
        if idx % 100 == 0: print(idx)
        each_edit_data = {}
        wrong_span = line['wrong_span']
        correct_span = line['correct_span']
        each_edit_data['text'] = line['left'] + line['wrong_span'] + line[
            'right']
        each_edit_data['correction'] = line['left'] + line['correct_span'] + \
                                       line['right']
        ##get word level edits of s label
        wrong = get_final_text_edits(each_edit_data['text'],
                                     each_edit_data['correction'])
        offset = len(line['left'])
        # print('wrong',wrong)

        each_edit_data['wrong'] = [[str(i[0] + 1), str(i[1] + 1), i[2]] for i in
                                   wrong]
        each_edit_data['wrong_no_offset'] = [
            [str(i[0] + 1 - offset), str(i[1] + 1 - offset), i[2]] for i in
            wrong]
        each_edit_data['correct_span'] = correct_span
        each_edit_data['wrong_span'] = wrong_span
        each_edit_data['left'] = line['left']
        each_edit_data['right'] = line['right']

        data_slabel_on_word_level.append(each_edit_data)

    with open(out_file, 'w', encoding='utf-8')as f:
        for line in data_slabel_on_word_level:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

        # each_edit_data['text'] = line['text']
        # each_edit_data['correction'] = line['correction']
        # if line['text'] == line['correction']:
        #     each_edit_data['wrong'] = []
        # # else:each_edit_data['wrong']=get_final_text_edits(line['wrong_span'],line['correct_span'])
        # else:
        #     each_edit_data['wrong'] = get_final_text_edits(each_edit_data['text'], each_edit_data['correction'])
        # each_edit_data['wrong'] = [[str(i[0] + 1), str(i[1] + 1), i[2]] for i in each_edit_data['wrong']]


def get_s_label_word_level_on_span_pair(raw_data_file, out_file):
    raw_data_after_ppl = []
    with open(raw_data_file, 'r', encoding='utf-8')as f:
        for line in f.readlines():
            raw_data_after_ppl.append(json.loads(line.strip()))

    data_slabel_on_word_level = []
    for idx, line in enumerate(raw_data_after_ppl):
        # for idx,line in enumerate(raw_data_after_ppl[:200]):
        if idx % 100 == 0: print(idx)
        each_edit_data = {}
        wrong_span = line['wrong_span']
        correct_span = line['correct_span']
        if len(wrong_span) != len(correct_span):
            continue
        each_edit_data['text'] = line['left'] + line['wrong_span'] + line[
            'right']
        each_edit_data['correction'] = line['left'] + line['correct_span'] + \
                                       line['right']
        ##get word level edits of s label
        # wrong=get_final_text_edits(each_edit_data['text'],each_edit_data['correction'])
        wrong = get_final_text_edits(wrong_span, correct_span)
        offset = len(line['left'])
        each_edit_data['wrong'] = [
            [str(i[0] + 1 + offset), str(i[1] + 1 + offset), i[2]] for i in
            wrong]
        each_edit_data['wrong_no_offset'] = [
            [str(i[0] + 1), str(i[1] + 1), i[2]] for i in wrong]
        each_edit_data['correct_span'] = correct_span
        each_edit_data['wrong_span'] = wrong_span
        each_edit_data['left'] = line['left']
        each_edit_data['right'] = line['right']

        data_slabel_on_word_level.append(each_edit_data)

    with open(out_file, 'w', encoding='utf-8')as f:
        for line in data_slabel_on_word_level:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    pass

def write_json2txt(total_data,filename):
    f=open(filename,'w',encoding='utf-8')
    for line in total_data:
        line["char_wrong"] = line["wrong"]
        f.write(json.dumps(line,ensure_ascii=False)+'\n')
    f.close()
def read_json_fromTxt(filename):
    total_data=[]
    f=open(filename,'r',encoding='utf-8')
    for idx,line in enumerate(f.readlines()):
        total_data.append(json.loads(line.strip()))
    return total_data
def get_mask_index(char_list,mask_window_size=8):
    start_index=0
    for index,each_char in enumerate(char_list):
        if each_char=='[MASK]':
            start_index=index
            break
    #不包括end_index
    end_index=start_index+mask_window_size
    return start_index,end_index
def get_edit_file(data_file_name,out_file_name='./data_processed_v1.txt', tag="r"):
    data=read_json_fromTxt(data_file_name)
    edit_data_dict_list=[]
    for idx,line in enumerate(data):
        if tag == "r":
            if line["wrong_span"] <= line["correct_span"]:
                continue
        if tag == "m":
            if line["wrong_span"] >= line["correct_span"]:
                continue
        if '[]'in line['wrong_span']:continue
        if idx%500==0:print(idx)
        #print(line)
        each_edit_data={}
        wrong_span=line['wrong_span'].replace('[UNK]','“')
        correct_span=line['correct_span']
        each_edit_data['text']=line['left']+wrong_span+line['right']
        each_edit_data['correction']=line['left']+correct_span+line['right']

        each_edit_data['correction']=each_edit_data['correction'].replace('[unused1]','')
        if correct_span==wrong_span:
            each_edit_data['wrong']=[]
        else:
            each_edit_data['wrong']=get_text_edits_on_token(wrong_span,correct_span)
        if len(each_edit_data['wrong'])==0:continue
        offset=len(line['left'])
        each_edit_data["wrong_no_offset"] = [[str(i[0]+1),str(i[1]+1),i[2]]for i in each_edit_data['wrong']]
        each_edit_data["left"] = line["left"]
        each_edit_data["right"] = line["right"]
        each_edit_data["correct_span"] = correct_span
        each_edit_data["wrong_span"] = wrong_span
        each_edit_data['wrong']=[[str(i[0]+offset+1),str(i[1]+offset+1),i[2]]for i in each_edit_data['wrong']]
        edit_data_dict_list.append(each_edit_data)
    write_json2txt(edit_data_dict_list,out_file_name)

def test_slabel():
    wrong_span = '人类不能'
    correct_span = '人们不会'
    wrong_span = '因为我们俩从小就是形成不合的一对所以她特意的请我当她的伴娘。'
    correct_span = '因为我们俩从小就是形影不离的一对所以她特意地请我当她的伴娘。'
    wrong_span = '我们都认为生命是最重要的，那应该要先解决饥饿问题。'
    correct_span = '我们都认为生命是最重要的，那就应该先解决饥饿问题。'
    wrong_span = '为因我'
    correct_span = '为因为我'
    wrong = get_final_text_edits(wrong_span, correct_span)
    wrong = [[[str(i[0] + 1), str(i[1] + 1), i[2]] for i in wrong]]
    print('wrong', wrong)
    print('/'.join(jieba.cut(wrong_span)))
    print('/'.join(jieba.cut(correct_span)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlm_version',
                        default='transformer')
    parser.add_argument('--dir_name',
                        default="./")

    args = parser.parse_args()

    mlm_version = args.mlm_version
    dir_name = args.dir_name
    filenames = []

    # selection
    tag = "s"
    threshold = 2
    version = f"{tag}label{threshold}.final"
    raw_data_file = f'{dir_name}/{mlm_version}.denoise'  # ppl过滤后的文件
    out_file_name = f'{dir_name}/{mlm_version}.word.{version}'
    filenames.append(out_file_name)
    # #两个完整句子算word level edits
    # get_s_label_word_level_on_whole_sent_pair(raw_data_file,out_file)
    # 只对两个span算word level edits
    get_s_label_word_level_on_span_pair(raw_data_file, out_file_name)
    # 数据去噪，目前只去除存在标点符号错误的情况，输出文件路径默认为 mid_file_name[:-4]+'_denoise_v2.txt'
    filter_noise_in_augment_data(out_file_name)


    # redundant
    start_time = time.time()
    tag = "r"
    threshold = 2
    version = f"{tag}label{threshold}.final"
    # data_file_name = f"{dir_name}/{mlm_version}.denoise.{version}"
    out_file_name = f"{dir_name}/{mlm_version}.word.{version}"
    filenames.append(out_file_name)

    get_edit_file(raw_data_file, out_file_name, tag)

    # missing
    tag = "m"
    threshold = 2
    # data_file_name = f"{dir_name}/{mlm_version}.denoise.{version}"
    version = f"{tag}label{threshold}.final"
    out_file_name = f"{dir_name}/{mlm_version}.word.{version}"
    filenames.append(out_file_name)

    get_edit_file(raw_data_file, out_file_name, tag)


    out_all_filename = f"{dir_name}/{mlm_version}.all"
    wf = open(out_all_filename, "w")
    for filename in filenames:
        with open(filename) as f:
            wf.write(f.read())
    wf.close()
    end_time = time.time()
    print(end_time - start_time)


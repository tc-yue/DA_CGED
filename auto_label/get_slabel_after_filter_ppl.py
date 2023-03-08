# -*- coding: utf-8 -*-
# @Time    : 2021/9/16 11:30
# @Author  : shengksong
# @FileName: get_slabel_after_filter_ppl.py
# @Software: PyCharm
import sys
# sys.path.append('../')
# from get_text_edits_on_processed_train_data_v4 import get_final_text_edits
from get_text_edits_on_processed_train_data import get_final_text_edits
import json
import jieba
def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False

def filter_not_chinese(text,wrong_list,filter_label='S'):
    #过滤掉特定类型的错误与非中文的错误
    # not_chinese_num=0
    for i in wrong_list:
        # if i[-1]!=filter_label or is_chinese(text[int(i[0])-1])==False:return False
        if is_chinese(text[int(i[0])-1])==False:
            return False
            # not_chinese_num+=1
    # print('not chinese num',not_chinese_num)
    return True
def filter_noise_in_augment_data(filename):
    raw_data_after_ppl=[]
    with open(filename,'r',encoding='utf-8')as f:
        for line in f.readlines():
            raw_data_after_ppl.append(json.loads(line.strip()))
    data_denoise=[]
    not_chinese_num=0
    for idx,line in enumerate(raw_data_after_ppl):
        if filter_not_chinese(line['text'],line['wrong'],'S'):
            data_denoise.append(line)
        else:
            not_chinese_num+=1
            # print('noise',line)
            continue
    print('not chinese num',not_chinese_num)
    with open(filename,'w',encoding='utf-8')as f:
        for line in data_denoise:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def get_s_label_word_level_on_whole_sent_pair(raw_data_file,out_file):
    raw_data_after_ppl=[]
    with open(raw_data_file,'r',encoding='utf-8')as f:
        for line in f.readlines():
            raw_data_after_ppl.append(json.loads(line.strip()))

    data_slabel_on_word_level=[]
    for idx,line in enumerate(raw_data_after_ppl):
    # for idx,line in enumerate(raw_data_after_ppl[:200]):
        if idx%100==0:print(idx)
        each_edit_data = {}
        wrong_span=line['wrong_span']
        correct_span=line['correct_span']
        each_edit_data['text']=line['left']+line['wrong_span']+line['right']
        each_edit_data['correction']=line['left']+line['correct_span']+line['right']
        ##get word level edits of s label
        wrong=get_final_text_edits(each_edit_data['text'],each_edit_data['correction'])
        offset=len(line['left'])
        # print('wrong',wrong)

        each_edit_data['wrong']=[[str(i[0]+1),str(i[1]+1),i[2]] for i in wrong]
        each_edit_data['wrong_no_offset']=[[str(i[0]+1-offset),str(i[1]+1-offset),i[2]]for i in wrong]
        each_edit_data['correct_span'] = correct_span
        each_edit_data['wrong_span'] = wrong_span
        each_edit_data['left']=line['left']
        each_edit_data['right']=line['right']

        data_slabel_on_word_level.append(each_edit_data)

    with open(out_file,'w',encoding='utf-8')as f:
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

def get_s_label_word_level_on_span_pair(raw_data_file,out_file):
    raw_data_after_ppl=[]
    with open(raw_data_file,'r',encoding='utf-8')as f:
        for line in f.readlines():
            raw_data_after_ppl.append(json.loads(line.strip()))

    data_slabel_on_word_level=[]
    for idx,line in enumerate(raw_data_after_ppl):
    # for idx,line in enumerate(raw_data_after_ppl[:200]):
        if idx%100==0:print(idx)
        each_edit_data = {}
        wrong_span=line['wrong_span']
        correct_span=line['correct_span']
        each_edit_data['text']=line['left']+line['wrong_span']+line['right']
        each_edit_data['correction']=line['left']+line['correct_span']+line['right']
        ##get word level edits of s label
        # wrong=get_final_text_edits(each_edit_data['text'],each_edit_data['correction'])
        wrong=get_final_text_edits(wrong_span,correct_span)
        offset=len(line['left'])
        each_edit_data['wrong']=[[str(i[0]+1+offset),str(i[1]+1+offset),i[2]] for i in wrong]
        each_edit_data['wrong_no_offset']=[[str(i[0]+1),str(i[1]+1),i[2]]for i in wrong]
        each_edit_data['correct_span'] = correct_span
        each_edit_data['wrong_span'] = wrong_span
        each_edit_data['left']=line['left']
        each_edit_data['right']=line['right']

        data_slabel_on_word_level.append(each_edit_data)

    with open(out_file,'w',encoding='utf-8')as f:
        for line in data_slabel_on_word_level:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    pass
if __name__ == '__main__':

    #test
    #{'text': '因为我们俩从小就是形成不合的一对所以她特意的请我当她的伴娘。', 'correction': '因为我们俩从小就是形影不离的一对所以她特意地请我当她的伴娘。', 'wrong': [['10', '14', 'S'], ['22', '22', 'S']]}
    wrong_span='人类不能'
    correct_span='人们不会'
    wrong_span='因为我们俩从小就是形成不合的一对所以她特意的请我当她的伴娘。'
    correct_span='因为我们俩从小就是形影不离的一对所以她特意地请我当她的伴娘。'
    wrong_span='我们都认为生命是最重要的，那应该要先解决饥饿问题。'
    correct_span='我们都认为生命是最重要的，那就应该先解决饥饿问题。'
    wrong_span='为因我'
    correct_span='为因为我'
    wrong = get_final_text_edits(wrong_span, correct_span)
    wrong=[[[str(i[0]+1),str(i[1]+1),i[2]] for i in wrong]]
    print('wrong',wrong)
    print('/'.join(jieba.cut(wrong_span)))
    print('/'.join(jieba.cut(correct_span)))
    #获得word level edits，
    mlm_version = f"cged_mlm_mse_context_noposweight_epochs10"
    # mlm_version = f"nlpcc_cged_mlm_mse_context_noposweight_epochs10"
    dir_name='/apdcephfs/share_1324356/tianchiyue/GrammaticalErrorCorrection/final_data/'
    raw_data_file=f'{dir_name}{mlm_version}.denoise.slabel2.final'#ppl过滤后的文件
    mid_file_name=f'{dir_name}{mlm_version}.word.slabel2.final'
    # #两个完整句子算word level edits
    # get_s_label_word_level_on_whole_sent_pair(raw_data_file,out_file)
    # 只对两个span算word level edits
    get_s_label_word_level_on_span_pair(raw_data_file, mid_file_name)
    #数据去噪，目前只去除存在标点符号错误的情况，输出文件路径默认为 mid_file_name[:-4]+'_denoise_v2.txt'
    filter_noise_in_augment_data(mid_file_name)
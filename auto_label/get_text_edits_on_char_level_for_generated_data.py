# -*- coding: utf-8 -*-
# @Time    : 2021/9/10 16:21
# @Author  : shengksong
# @FileName: get_text_edits_on_char_level.py
# @Software: PyCharm

import json
import time
from get_text_edits_on_char_level import get_text_edits_on_token

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
def get_edit_file(data_file_name,out_file_name='./data_processed_v1.txt'):
    data=read_json_fromTxt(data_file_name)
    edit_data_dict_list=[]
    for idx,line in enumerate(data):
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
if __name__ == '__main__':
    start_time=time.time()
    tag = "r"
    threshold = 2
    mlm_version = f"cged_mlm_mse_context_noposweight_epochs10"
    # mlm_version = f"nlpcc_cged_mlm_mse_context_noposweight_epochs10"

    version = f"{tag}label{threshold}.final"
    data_file_name = f"/apdcephfs/share_1324356/tianchiyue/GrammaticalErrorCorrection/final_data/{mlm_version}.denoise.{version}"
    out_file_name = f"/apdcephfs/share_1324356/tianchiyue/GrammaticalErrorCorrection/final_data/{mlm_version}.word.{version}"
    get_edit_file(data_file_name,out_file_name)

    tag = "m"
    threshold = 2
    version = f"{tag}label{threshold}.final"
    data_file_name = f"/apdcephfs/share_1324356/tianchiyue/GrammaticalErrorCorrection/final_data/{mlm_version}.denoise.{version}"
    out_file_name = f"/apdcephfs/share_1324356/tianchiyue/GrammaticalErrorCorrection/final_data/{mlm_version}.word.{version}"
    get_edit_file(data_file_name, out_file_name)


    end_time=time.time()
    print(end_time-start_time)

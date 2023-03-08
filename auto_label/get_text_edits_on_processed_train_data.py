# -*- coding: utf-8 -*-
# @Time    : 2021/9/3 18:20
# @Author  : shengksong
# @FileName: get_text_edits_on_processed_train_data.py
# @Software: PyCharm

import json
import jieba
import copy
def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False
def read_json_fromTxt(filename):
    total_data=[]
    f=open(filename,'r',encoding='utf-8')
    for idx,line in enumerate(f.readlines()):
        # print(line)
        # if idx%20==0:print(idx)
        # if len(line)>=1894:print(line[1892],line[1893],line[1894])
        total_data.append(json.loads(line.strip()))
    return total_data
def minDistance(word1, word2) -> int:
    if len(word1) == 0:
        return len(word2)
    elif len(word2) == 0:
        return len(word1)
    M = len(word1)
    N = len(word2)
    output = [[0] * (N + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                output[i][j] = 0
            elif i == 0 and j != 0:
                output[i][j] = j
            elif i != 0 and j == 0:
                output[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                output[i][j] = output[i - 1][j - 1]
            else:
                output[i][j] = min(output[i - 1][j - 1] + 1, output[i - 1][j] + 1, output[i][j - 1] + 1)
    # for i in output:print(i)
    return output

def backtrackingPath(word1,word2):
    dp = minDistance(word1,word2)
    m = len(dp)-1
    n = len(dp[0])-1
    operation_list=[[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m+1):operation_list[i][0]='@'
    for i in range(n+1):operation_list[0][i]='@'
    operation = []
    spokenstr = []
    writtenstr = []

    while n>=0 or m>=0:
        if n and dp[m][n-1]+1 == dp[m][n]:
            operation_list[m][n]='in'
            print("insert %c\n" %(word2[n-1]))
            spokenstr.append("insert")
            writtenstr.append(word2[n-1])
            # operation.append("NULLREF:"+word2[n-1])
            operation.append("insert:"+word2[n-1])
            n -= 1
            continue
        if m and dp[m-1][n]+1 == dp[m][n]:
            operation_list[m][n]='de'
            print("delete %c\n" %(word1[m-1]))
            spokenstr.append(word1[m-1])
            writtenstr.append("delete")
            # operation.append(word1[m-1]+":NULLHYP")
            operation.append(word1[m-1]+":delete")
            m -= 1
            continue
        if dp[m-1][n-1]+1 == dp[m][n]:
            operation_list[m][n]='su'
            print("replace %c %c\n" %(word1[m-1],word2[n-1]))
            spokenstr.append(word1[m - 1])
            writtenstr.append(word2[n-1])
            # operation.append(word1[m - 1] + ":"+word2[n-1])
            operation.append('replace'+word1[m - 1] + ":"+word2[n-1])
            n -= 1
            m -= 1
            continue
        if dp[m-1][n-1] == dp[m][n]:
            operation_list[m][n]='no'
            spokenstr.append(' ')
            writtenstr.append(' ')
            operation.append(word1[m-1])
        n -= 1
        m -= 1
    spokenstr = spokenstr[::-1]
    writtenstr = writtenstr[::-1]
    operation = operation[::-1]
    # print(spokenstr,writtenstr)
    # print(operation)
    # for i in operation_list:print(i)
    # print('*****')
    return spokenstr,writtenstr,operation
def backtrackingPath2(word1,word2):
    #替换优先
    dp = minDistance(word1,word2)
    m = len(dp)-1
    n = len(dp[0])-1
    operation_list=[[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m+1):operation_list[i][0]='@'
    for i in range(n+1):operation_list[0][i]='@'
    operation = []
    only_operation_list=[]
    spokenstr = []
    writtenstr = []

    while n>=0 or m>=0:
        # print('aa',operation)
        # print('bb',only_operation_list)
        if dp[m-1][n-1]+1 == dp[m][n]:
            operation_list[m][n]='su'
            # print("replace %c %c\n" %(word1[m-1],word2[n-1]))
            spokenstr.append(word1[m - 1])
            writtenstr.append(word2[n-1])
            # operation.append(word1[m - 1] + ":"+word2[n-1])
            operation.append('su:'+word1[m - 1] + ":"+word2[n-1])
            only_operation_list.append('S')
            n -= 1
            m -= 1
            continue
        if n and dp[m][n-1]+1 == dp[m][n]:
            operation_list[m][n]='in'
            # print("insert %c\n" %(word2[n-1]))
            spokenstr.append("insert")
            writtenstr.append(word2[n-1])
            # operation.append("NULLREF:"+word2[n-1])
            operation.append("M:"+word2[n-1])
            only_operation_list.append('M')
            n -= 1
            continue
        if m and dp[m-1][n]+1 == dp[m][n]:
            operation_list[m][n]='de'
            # print("delete %c\n" %(word1[m-1]))
            spokenstr.append(word1[m-1])
            writtenstr.append("delete")
            # operation.append(word1[m-1]+":NULLHYP")
            operation.append(word1[m-1]+":de")
            only_operation_list.append('R')
            m -= 1
            continue
        if dp[m-1][n-1] == dp[m][n]:
            operation_list[m][n]='no'
            spokenstr.append(' ')
            writtenstr.append(' ')
            operation.append(word1[m-1])
            only_operation_list.append('')
        n -= 1
        m -= 1
    spokenstr = spokenstr[::-1]
    writtenstr = writtenstr[::-1]
    operation = operation[::-1]
    only_operation_list=only_operation_list[::-1]
    return spokenstr,writtenstr,operation,only_operation_list
def get_start_index_and_end_index_from_jieba(index,sent,operation_list,print_flag=False):
    word_list=list(jieba.cut(sent))
    if print_flag:print('/'.join(word_list))
    current_length=0#index从1开始
    for each_word in word_list:
        start_index=current_length
        current_length=current_length+len(each_word)
        end_index=current_length-1
        count_num = count_missing_position_before_this_index(operation_list, index)
        new_index=index-count_num
        if start_index<=new_index and end_index>=new_index:
            return start_index,end_index
    return index,index
def count_missing_position_before_this_index(operation_list,wrong_index):
    count_num=0
    for idx,i in enumerate(operation_list):
       # if idx <wrong_index-1 and i=='M':count_num+=1
       if idx <wrong_index and i=='M':count_num+=1
    return count_num
# def get_wrong_order_error_for_aligned_seqs(a,b,a_list,b_list,diff_word_index_list):
def get_wrong_order_error_for_aligned_seqs(a,b):
    '''
    为能够按照词级别对齐的句子，找出wrong_order错误，默认一句中最多一个wrong_order
    '''
    if len(a)!=len(b):return []
    else:
        window_size_list=[i for i in range(1,len(a)+1)]
        for each_window in window_size_list:
            for index in range(0,len(a)-each_window+1):
                a_sub_str=a[index:index+each_window+1]
                b_sub_str=b[index:index+each_window+1]
                if sorted(a_sub_str)==sorted(b_sub_str)and a_sub_str!=b_sub_str:
                    wrong_list=[[index,index+each_window,'W']]
                    return wrong_list
        return []


def get_alignment_between_two_seq(a,b):
    '''
    暂时只处理存在一个词不一致的情况
    :param a:
    :param b:
    :return:
    '''
    alignment_flag=False
    wrong_list=[]
    a_list=list(jieba.cut(a))
    b_list=list(jieba.cut(b))
    if len(a_list)!=len(b_list):
        return alignment_flag,wrong_list
    else:
        alignment_flag=True
        if len(a)==len(b):
            wrong_list=get_wrong_order_error_for_aligned_seqs(a,b)
            if len(wrong_list)>0:return alignment_flag,wrong_list
        #wrong_order没有返回结果，再走下面的
        #首先记录不一样的词的的index的数目
        diff_word_num=0
        diff_word_index_list=[]
        for index in range(len(a_list)):
            if a_list[index]!=b_list[index]:
                diff_word_num+=1
                diff_word_index_list.append(index)
        # print(a,b)
        # print('diff_list',diff_word_index_list)
        # if diff_word_num==1:#暂时只处理存在一个词不同的情况
        #wrong_order的级别优先
        # if
        if diff_word_num==1 and len(a_list[diff_word_index_list[0]])==len(b_list[diff_word_index_list[0]]):#暂时只处理存在一个词不同的情况
        # if diff_word_num==1 and len(a_list[diff_word_index_list[0]])<=len(b_list[diff_word_index_list[0]]):#暂时只处理存在一个词不同的情况
            for each_diff_word_index in diff_word_index_list:
                start_index=0
                for each_index,each_word in enumerate(a_list):
                    if each_index<each_diff_word_index:start_index+=len(each_word)
                end_index=start_index+len(a_list[each_diff_word_index])-1
                wrong_list.append([start_index,end_index,'S'])
            return alignment_flag, wrong_list
        elif diff_word_num==2 and diff_word_index_list[0]+1==diff_word_index_list[1]:
            if sorted(a_list[diff_word_index_list[0]]+a_list[diff_word_index_list[1]])\
                    ==sorted(b_list[diff_word_index_list[0]]+b_list[diff_word_index_list[1]]):
                #两个词乱序的情况
                start_index=0
                for each_index,each_word in enumerate(a_list):
                    if each_index<diff_word_index_list[0]:start_index+=len(each_word)
                end_index=start_index+len(a_list[diff_word_index_list[0]])+len(a_list[diff_word_index_list[1]])-1
                wrong_list.append([start_index,end_index,'W'])
                # print('alignment')
                # print(a,b)
                # print('/'.join(a_list),'/'.join(b_list))
                # print(alignment_flag,wrong_list)
            return alignment_flag,wrong_list
            # else:
            #     for each_diff_word_index in diff_word_index_list:
            #         start_index = 0
            #         for each_index, each_word in enumerate(a_list):
            #             if each_index < each_diff_word_index: start_index += len(each_word)
            #         end_index = start_index + len(a_list[each_diff_word_index]) - 1
            #         wrong_list.append([start_index, end_index, 'S'])
            # return alignment_flag, wrong_list
        elif diff_word_num==3 and diff_word_index_list[0]+1==diff_word_index_list[1] and diff_word_index_list[1]+1==diff_word_index_list[2]:
            if sorted(a_list[diff_word_index_list[0]]+a_list[diff_word_index_list[1]]+a_list[diff_word_index_list[2]])\
                    ==sorted(b_list[diff_word_index_list[0]]+b_list[diff_word_index_list[1]]+b_list[diff_word_index_list[2]]):
                #两个词乱序的情况
                start_index=0
                for each_index,each_word in enumerate(a_list):
                    if each_index<diff_word_index_list[0]:start_index+=len(each_word)
                end_index=start_index+len(a_list[diff_word_index_list[0]])+len(a_list[diff_word_index_list[1]])+len(a_list[diff_word_index_list[2]])-1
                wrong_list.append([start_index,end_index,'W'])
                # return alignment_flag,wrong_list
            return alignment_flag,wrong_list
        else:
            return False,[]

            # for index,each_word in enumerate(a_list):#可能有多个，这里先写只有一个的情况
            #     a_str=''.join(a_list[:index]+a_list[index+1:])#去除改字之后，比较其他的部分
            #     b_str=''.join(b_list[:index]+b_list[index+1:])
            #     if a_str==b_str:
            #         alignment_flag=True
            #         start_index=len(''.join(a_list[:index]))
            #         end_index=start_index+len(each_word)-1
            #         wrong_list=[start_index,end_index,'S']
            #         return alignment_flag,wrong_list

def post_process_wrong_list(wrong_list,sent):
    '''
    处理下面这种类型：
    873 这样就会造成更多的人因缺少粮食而挨饿。 这样/就/会/造成/更/多/的/人/因/缺少/粮食/而/挨饿/。
    873 这样就会使更多的人因缺少粮食而挨饿。 这样/就/会/使/更/多/的/人/因/缺少/粮食/而/挨饿/。
    gold [[4, 5, 'S']]
    pred [[4, 4, 'R'], [4, 5, 'S']]index之间存在重复
    *****
    94 所以我认为世界上所有的自杀者也可以起诉法院，和别的刑事案件不一样的地方是受害者和被害者是同一个人。 所以/我/认为/世界/上/所有/的/自杀者/也/可以/起诉/法院/，/和/别的/刑事案件/不/一样/的/地方/是/受害者/和/被害者/是/同一个/人/。
    94 所以我认为世界上所有的自杀者也可以起诉法院，和别的刑事案件不一样的地方是受害者和罪犯是同一个人。 所以/我/认为/世界/上/所有/的/自杀者/也/可以/起诉/法院/，/和/别的/刑事案件/不/一样/的/地方/是/受害者/和/罪犯/是/同一个/人/。
    gold [[40, 42, 'S']]
    pred [[40, 40, 'R'], [40, 42, 'S']]
    :return:
    '''
    for i in wrong_list:
        if i[0] == i[1] and i[2] == 'S' and is_chinese(sent[i[0]]) == False:
            wrong_list.remove(i)
    processed_wrong_list=copy.deepcopy(wrong_list)
    # print(processed_wrong_list)
    if len(wrong_list)<2:return wrong_list
    # for index in range(len(wrong_list)-1):
    for index,i in enumerate(wrong_list[:-1]):
        if wrong_list[index][0]==wrong_list[index+1][0]:
            # wrong_list.remove(i)
            processed_wrong_list.remove(i)
        else:continue
    return processed_wrong_list


def post_process_operation_list(operation_list,sent,new_sent,print_flag=False):
    # print(sent)
    # print(new_sent)
    alignment_flag,wrong_list=get_alignment_between_two_seq(sent,new_sent)
    if alignment_flag and len(wrong_list)>0:
        # print('alignment function:',sent)
        return wrong_list
    else:
        wrong_list=[]
        for index,i in enumerate(operation_list):
            wrong_index=index
            if i=='':continue
            elif i=='S':#当存在错别字的情况时，按照词力度判断index
                start_index,end_index=get_start_index_and_end_index_from_jieba(wrong_index,sent,operation_list,print_flag)
                wrong_list.append((start_index,end_index,i))
            #新加的部分，当'R','M'错误时，也需要考虑前面其他的'M'符号
            elif i=='R' or i=='M':
                count_num=count_missing_position_before_this_index(operation_list,wrong_index)
                wrong_list.append((wrong_index-count_num,wrong_index-count_num,i))
            else:
                wrong_list.append((wrong_index,wrong_index,i))
        #去重
        wrong_list_no_repetation=list(set(wrong_list))
        wrong_list_no_repetation.sort(key=wrong_list.index)
        wrong_list_no_repetation=[list(i)for i in wrong_list_no_repetation]

        #连续错误，可能需要合并成一个
        for i in wrong_list_no_repetation:
            if i[2]=='M'and [i[0]+1,i[1]+1,'M']in wrong_list_no_repetation:
                wrong_list_no_repetation.remove([i[0]+1,i[1]+1,'M'])
        for i in wrong_list_no_repetation:
            #可能存在三个R连在一起的情况，
            if i[2]=='R'and[i[0]+1,i[1]+1,'R']in wrong_list_no_repetation and [i[0]+2,i[1]+2,'R'] in wrong_list_no_repetation:
                wrong_list_no_repetation.remove([i[0]+1,i[1]+1,'R'])
                wrong_list_no_repetation.remove([i[0]+2,i[1]+2,'R'])
                wrong_list_no_repetation.remove(i)
                wrong_list_no_repetation.append([i[0],i[1]+2,'R'])
            elif i[2]=='R'and [i[0]+1,i[1]+1,'R']in wrong_list_no_repetation:
                wrong_list_no_repetation.remove([i[0]+1,i[1]+1,'R'])
                wrong_list_no_repetation.remove(i)
                wrong_list_no_repetation.append([i[0],i[1]+1,'R'])
        # for index,i in enumerate(wrong_list_no_repetation):
        #     if i[2]=='S' and int(i[0])<int(i[1]) :
        #         if set(sent[int(i[0])-1:int(i[1])])==set(new_sent[int(i[0])-1:int(i[1])]):
        #             print('here')
        #             i=[i[0],i[1],'W']
        #             wrong_list_no_repetation[index]=i
        wrong_list_no_repetation=post_process_wrong_list(wrong_list_no_repetation,sent)
        # wrong_list_no_repetation=[[str(i[0]+1),str(i[1]+1),i[2]]for i in wrong_list_no_repetation]
    return wrong_list_no_repetation
def get_final_text_edits(a,b)->list:
    # backtrackingPath2(a,b)
    spokenstr, writtenstr, operation, operation_list = backtrackingPath2(a, b)
    edit_list = post_process_operation_list(operation_list, a, b, print_flag=True)
    # edit_list=[[str(i[0]+1),str(i[1]+1),i[2]]for i in edit_list]
    return edit_list
def get_jieba_cut_result(a):
    a_list=jieba.cut(a)
    # b_list=jieba.cut(b)
    a_jieba_cut='/'.join(a_list)
    # b_jieba_cut='/'.join(b_list)
    return a_jieba_cut
if __name__ == '__main__':

    '''
    {"year": "2016", "text": "对我们国家来说，帮挨饿的人是当然做的事情。", 
    "correction": "对我们国家来说，帮挨饿的人是当然要做的事情。", 
    "wrong": [["17", "17", "M"]]}

    '''
    wrong_dict_list=['真不知道','举子','这件','一件','来说','来看','更深','太夜','走白','喜喜','看过','挑早','的话','答到']
    for i in wrong_dict_list:jieba.del_word(i)
    a='我吃饭'
    b='吃饭'
    # a='吃饭'
    # b='我吃饭'

    a='根据一件事情来说，这位妻子知道了自己得了一种不治之症。'
    b='根据这件事情来说，这位妻子知道了自己得了一种不治之症。'
    a='这样给爸妈写信可能是第一次的。'
    b= '这样给爸妈写信可能是第一次。'
    a='抽烟的习惯对城市环境也有不好处。'
    b='抽烟的习惯对城市环境也有坏处。'
    a='于是她的爸爸同意女儿的爱好。'
    b='但是她的爸爸同意女儿的爱好。'
    a='我认为男女分班给少年男女带来的短处比长处还多。'
    b='我认为男女分班给少年男女带来的坏处比好处还多。'
    a='但是最近大家开始注重吸烟的负方面。'
    b='但是最近大家开始注重吸烟的负影响。'
    a='真是不得不选的最后一个想法。在短文上的夫妻如此。'
    b='真是不得不选的最后一个办法。在短文里的夫妻就是如此。'
    a='瘦了也不满，发福也不满，真不知道。'
    b='瘦了也不满，发福也不满，真不明白。'
    a='在我成长的过程中不知多少次打我、指责我，我并不生气，我已经深深的理解到那都是为我好。'
    b='在我成长的过程中你们不知多少次打我、指责我，我并不生气，我已经深深地理解到那都是为我好。'
    a='我明天休息，喝酒，然后大睡觉。哈哈。'
    b='我明天休息，打算喝酒，然后睡大觉。哈哈。'
    a='我们每天不能互相见面。'
    b='我们每天不能互相见面。'

    a='还是先我们都用日语对话，然后用汉语吗？'
    b='还是我们先都用日语对话，然后再用汉语呢？'
    # a='到底是健康重要，还是粮食生产量重要呢？对这个问题我意见是产生量更重要。'
    # b='到底是健康重要，还是粮食生产量重要呢？对这个问题我的意见是产生量更重要。'
    a='这样就会造成更多的人因缺少粮食而挨饿。'
    b='这样就会使更多的人因缺少粮食而挨饿。'
    '''
        873 这样就会造成更多的人因缺少粮食而挨饿。 这样/就/会/造成/更/多/的/人/因/缺少/粮食/而/挨饿/。
    873 这样就会使更多的人因缺少粮食而挨饿。 这样/就/会/使/更/多/的/人/因/缺少/粮食/而/挨饿/。
    '''
    a='可是，问我选出一本最喜欢读的书，我就很为难了。可是，我还是会全力以赴，尽量答到这道题。'
    b='可是，让我选出一本最喜欢读的书，我就很为难了。可是，我还是会全力以赴，尽量答出这道题。'
    a='自己的孩子，你是青少年'
    b='自己的孩子：如果你是青少年'
    a='不仅是病者本，于病者的家人和朋友，都是一件难受的事。'
    b='不仅是病人本身，对于病者的家人和朋友，都是一件难受的事。'

    a='如一个长期受着病魔折磨的人活着'
    b='如果一个长期受着病魔折磨的人活着'
    spokenstr, writtenstr, operation,operation_list=backtrackingPath2(a,b)
    wrong_list=post_process_operation_list(operation_list,a,b,print_flag=True)
    print(a)
    print(b)
    print(wrong_list)



    spokenstr, writtenstr, operation,operation_list=backtrackingPath2(a,b)
    print(spokenstr)
    print(writtenstr)
    print(operation)
    print(len(a),len(operation_list),operation_list)
    wrong_list=post_process_operation_list(operation_list,a,b,print_flag=True)
    print('wrong list',wrong_list)

    evaluation_flag=False
    case_study_flag=True

    evaluation_flag=True
    case_study_flag=False
    if not evaluation_flag:
        pass
    else:
        # total_data=read_json_fromTxt('../../data/CGED_Data/train_processed_processed.txt')
        total_data=read_json_fromTxt('./data/train_processed_processed.txt')
        total_data=read_json_fromTxt('./data/train_processed.txt')
        total_gold_error=0.
        total_pred_error=0.
        total_pred_right_error=0.
        for index,i in enumerate(total_data):
            if index%100==0:print(index)
            a=i['text']
            b=i['correction']
            if a==''or b=='':continue
            # print(index,a,b)
            spokenstr, writtenstr, operation, operation_list = backtrackingPath2(a, b)
            wrong_list=post_process_operation_list(operation_list,a,b)
            wrong_list=[[str(i[0]+1),str(i[1]+1),i[2]]for i in wrong_list]
            gold_wrong_list=i['wrong']
            total_gold_error+=len(gold_wrong_list)
            total_pred_error+=len(wrong_list)
            each_pred_right_num=len([i for i in wrong_list if i in gold_wrong_list])
            total_pred_right_error+=each_pred_right_num

            if case_study_flag:
                if sorted(gold_wrong_list)!=sorted(wrong_list) and len(list(jieba.cut(a)))==len(list(jieba.cut(b))):#对于词粒度个数相等，可以直接对齐的情况，debug
                # if gold_wrong_list!=wrong_list:
                    print(index,a,get_jieba_cut_result(a))
                    print(index,b,get_jieba_cut_result(b))
                    print('gold',gold_wrong_list)
                    print('pred',wrong_list)

        p=total_pred_right_error/(total_pred_error+0.001)
        r=total_pred_right_error/(total_gold_error+0.001)
        f=2*p*r/(p+r+0.001)
        print('total gold error',total_gold_error)
        print('total pred error',total_pred_error)
        print('total pred right error',total_pred_right_error)
        print('p:',p)
        print('r:',r)
        print('f:',f)

# -*- coding: utf-8 -*-
# @Time    : 2021/9/3 18:20
# @Author  : shengksong
# @FileName: get_text_edits_on_processed_train_data.py
# @Software: PyCharm

import json
import jieba
import copy
def get_diff_word_num_and_list(a_list,b_list):
    #从jieba分词后的列表中，获取diff的word的数目与word_index_list
    diff_word_num = 0
    diff_word_index_list = []
    for index in range(len(a_list)):
        if a_list[index] != b_list[index]:
            diff_word_num += 1
            diff_word_index_list.append(index)
    return diff_word_num,diff_word_index_list
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
            # operation.append('su:'+word1[m - 1] + ":"+word2[n-1])
            operation.append(['S',word1[m - 1],word2[n-1]])
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
            # operation.append("M:"+word2[n-1])
            operation.append(['M',word2[n-1]])
            only_operation_list.append('M')
            n -= 1
            continue
        if m and dp[m-1][n]+1 == dp[m][n]:
            operation_list[m][n]='de'
            # print("delete %c\n" %(word1[m-1]))
            spokenstr.append(word1[m-1])
            writtenstr.append("delete")
            # operation.append(word1[m-1]+":NULLHYP")
            # operation.append('de:'+word1[m-1])
            operation.append(['R',word1[m-1]])
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
    # if print_flag:print('/'.join(word_list))
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
def process_when_wrong_order_span_is_a_complete_word(a,b,wrong_order_list):
    a_list=list(jieba.cut(a))
    for index,i in enumerate(wrong_order_list):
        wrong_order_span=a[i[0]:i[1]+1]
        if wrong_order_span in a_list:
            # wrong_order_list.remove(i)
            wrong_order_list[index]=[i[0],i[1],'S']#对于是一个整词的情况，将'W'替换为'S'
    return wrong_order_list


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
        diff_word_num,diff_word_index_list=get_diff_word_num_and_list(a_list,b_list)
        #wrong_order的级别优先

        #修改了下面的逻辑，可以对齐，且diff_num=1的情况下，直接置为'S'
        if diff_word_num==1 and len(a_list[diff_word_index_list[0]])==len(b_list[diff_word_index_list[0]]):#暂时只处理存在一个词不同的情况
        # if diff_word_num==1 and len(a_list[diff_word_index_list[0]])<=len(b_list[diff_word_index_list[0]]):#暂时只处理存在一个词不同的情况
            for each_diff_word_index in diff_word_index_list:
                start_index=0
                for each_index,each_word in enumerate(a_list):
                    if each_index<each_diff_word_index:start_index+=len(each_word)
                end_index=start_index+len(a_list[each_diff_word_index])-1
                wrong_list.append([start_index,end_index,'S'])
            return alignment_flag, wrong_list
        # if diff_word_num==1:
        #     for each_diff_word_index in diff_word_index_list:
        #         start_index=0
        #         for each_index,each_word in enumerate(a_list):
        #             if each_index<each_diff_word_index:start_index+=len(each_word)
        #         end_index=start_index+len(a_list[each_diff_word_index])-1
        #         wrong_list.append([start_index,end_index,'S'])
        #     return alignment_flag, wrong_list
        elif diff_word_num==2 and diff_word_index_list[0]+1==diff_word_index_list[1]:#两个diff word，并且两个diff word是连着的
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
                #三个词乱序的情况
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

def post_process_wrong_list(wrong_list,sent,new_sent):
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
    # print(sent)
    # print(new_sent)
    # print(wrong_list)
    for i in wrong_list:
        if i[0] == i[1] and i[2] == 'S' and is_chinese(sent[i[0]]) == False:
            wrong_list.remove(i)
    processed_wrong_list=copy.deepcopy(wrong_list)
    # print(processed_wrong_list)
    # if len(wrong_list)<2:return wrong_list
    # for index in range(len(wrong_list)-1):
    for index,i in enumerate(wrong_list[:-1]):
        if wrong_list[index][0]==wrong_list[index+1][0]:
            # wrong_list.remove(i)
            processed_wrong_list.remove(i)
        else:continue
    #处理结巴切分后的词表中，两词表可以对齐（即长度相等的情况）下，'S'错误容易误识别为'M','R'
    '''
    1157 如果我得了不治之症的话，我决定取安乐死。 如果/我/得/了/不治之症/的/话/，/我/决定/[15]取/安乐死/。
    1157 如果我得了不治之症的话，我决定采取安乐死。 如果/我/得/了/不治之症/的/话/，/我/决定/采取/安乐死/。
    gold [[15, 15, 'S']]
    pred [[15, 15, 'M']]
    780 我认为，人生的意义还是在于快乐中，并不是在痛苦中。 我/认为/，/人生/的/意义/还是/在于/快乐/中/，/并/不是/在/痛苦/中/。
    780 我认为，人生的意义还是在快乐中，并不是在痛苦中。 我/认为/，/人生/的/意义/还是/在/快乐/中/，/并/不是/在/痛苦/中/。
    gold [[11, 12, 'S']]
    pred [[12, 12, 'R']]
    '''
    ##基于case，加了下面那部分，反而指标下降了，不要了
    # a_list=list(jieba.cut(a))
    # b_list=list(jieba.cut(b))
    # # print('debug',wrong_list)
    # # print('/'.join(a_list))
    # # print('/'.join(b_list))
    # if len(a_list)!=len(b_list):return processed_wrong_list
    # else:
    #     for index,i in enumerate(processed_wrong_list):
    #         if i[-1]=='M':
    #             position_index=i[1]-1#i[1]-1
    #             current_length = 0  # index从1开始
    #             for each_word in a_list:
    #                 start_index = current_length
    #                 current_length = current_length + len(each_word)
    #                 end_index = current_length - 1
    #                 if start_index<=position_index and position_index<=end_index:
    #                     processed_wrong_list[index]=[start_index,end_index,'S']
    #                     break
    #         elif i[-1]=='R':
    #             position_index=i[1]-1# i[1]-1
    #             current_length = 0  # index从1开始
    #             for each_word in a_list:
    #                 start_index = current_length
    #                 current_length = current_length + len(each_word)
    #                 end_index = current_length - 1
    #                 if start_index<=position_index and position_index<=end_index:
    #                     processed_wrong_list[index]=[start_index,end_index,'S']
    #                     break
    # print('processed_wrong_list:',processed_wrong_list)
    '''
    处理下面这种格式
    14733 姆姆您好！ 姆/姆/您好/！
    14733 妈妈您好！ 妈妈/您好/！
    gold [[0, 1, 'S']]
    pred [[0, 0, 'S'], [1, 1, 'S']]
    14747 因为那座城市里奇峰彼起此伏，真是人间奇景。 因为/那座/城市/里/奇峰/彼起/此伏/，/真是/人间/奇景/。
    14747 因为那座城市里奇峰此起彼伏，真是人间奇景。 因为/那座/城市/里/奇峰/此起彼伏/，/真是/人间/奇景/。
    gold [[9, 12, 'S']]
    pred [[9, 10, 'S'], [11, 12, 'S']]
    '''
    processed_wrong_list2=copy.deepcopy(processed_wrong_list)
    # print(processed_wrong_list)
    for index,i in enumerate(processed_wrong_list):
        # print(index,i,sent)
        if i[-1]=='S'and i[0]==i[1] and [i[0]+1,i[0]+1,'S']in processed_wrong_list:
            try:
                processed_wrong_list2.remove(i)
                processed_wrong_list2.remove([i[0]+1,i[0]+1,'S'])
                processed_wrong_list2.append([i[0],i[0]+1,'S'])
            except:pass
        elif i[-1]=='S' and i[0]+1==i[1]and [i[1]+1,i[1]+2,'S']in processed_wrong_list:
            try:
                processed_wrong_list2.remove(i)
                processed_wrong_list2.remove([i[1]+1,i[1]+2,'S'])
                processed_wrong_list2.append([i[0],i[1]+3,'S'])
            except:pass
    # print('processed wrong_list',processed_wrong_list2)
    # # 多轮处理，知道不再发生变化
    # if  sorted(processed_wrong_list2)!=sorted(post_process_wrong_list(processed_wrong_list2,sent,new_sent)):
    #     # print(sorted(processed_wrong_list2),sorted(post_process_wrong_list(processed_wrong_list2, sent, new_sent)))
    #     processed_wrong_list2=post_process_wrong_list(processed_wrong_list2,sent,new_sent)
    #     # if processed_wrong_list2
    return processed_wrong_list2

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
                    print(wrong_list)
                    # wrong_list=process_when_wrong_order_span_is_a_complete_word(a,b,wrong_list)
                    wrong_list=[[i,i,'W']for i in range(wrong_list[0][0],wrong_list[0][1]+1)]
                    return wrong_list
        return []
def post_process_operation_list_on_token_level(operation_list,sent,new_sent,print_flag=False):
    wrong_list = []
    for index, i in enumerate(operation_list):
        wrong_index = index
        if i == '':
            continue
        elif i == 'S':  # 当存在错别字的情况时，按照词力度判断index
            count_num = count_missing_position_before_this_index(operation_list, index)
            new_index = index - count_num
            wrong_list.append((new_index, new_index, i))
        # 新加的部分，当'R','M'错误时，也需要考虑前面其他的'M'符号
        elif i == 'R' or i == 'M':
            count_num = count_missing_position_before_this_index(operation_list, wrong_index)
            wrong_list.append((wrong_index - count_num, wrong_index - count_num, i))
        else:
            wrong_list.append((wrong_index, wrong_index, i))
    # print('before quchong',wrong_list)
    # 去重
    wrong_list_no_repetation = list(set(wrong_list))
    wrong_list_no_repetation.sort(key=wrong_list.index)
    wrong_list_no_repetation = [list(i) for i in wrong_list_no_repetation]
    #去除标点符号
    ori_wrong_list = copy.deepcopy(wrong_list_no_repetation)
    # for i in ori_wrong_list:
    #     if i[0]>0 and is_chinese(sent[i[0]-1]) == False:wrong_list_no_repetation.remove(i)
    if len(sent)==len(new_sent):
        # print('equal len')
        wrong_order_list=get_wrong_order_error_for_aligned_seqs(sent,new_sent)
        if len(wrong_order_list)>0:
            # print(wrong_list_no_repetation)
            for i in ori_wrong_list:
                # print('i',i)
                if [i[0],i[1],'W'] in wrong_order_list:
                    # print('here')
                    wrong_list_no_repetation.remove(i)
            wrong_list_no_repetation=wrong_list_no_repetation+wrong_order_list
            wrong_list_no_repetation=sorted(wrong_list_no_repetation)

    return wrong_list_no_repetation
def post_process_operation_list(operation_list,sent,new_sent,print_flag=False):
    # print(sent)
    # print(new_sent)
    alignment_flag,wrong_list=get_alignment_between_two_seq(sent,new_sent)
    if alignment_flag and len(wrong_list)>0:
        # print('alignment function:',sent)
        return wrong_list
    else:
        # print('debug 2')
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
            #分别处理可能存在四个，三个，两个R连在一起的情况，
            if i[2]=='R'and [i[0]+1,i[1]+1,'R']in wrong_list_no_repetation and [i[0]+2,i[1]+2,'R'] in wrong_list_no_repetation and [i[0]+3,i[1]+3,'R']in wrong_list_no_repetation:
                wrong_list_no_repetation.remove([i[0]+1,i[1]+1,'R'])
                wrong_list_no_repetation.remove([i[0]+2,i[1]+2,'R'])
                wrong_list_no_repetation.remove([i[0]+3,i[1]+3,'R'])
                wrong_list_no_repetation.remove(i)
                wrong_list_no_repetation.append([i[0],i[1]+3,'R'])
            elif i[2]=='R'and[i[0]+1,i[1]+1,'R']in wrong_list_no_repetation and [i[0]+2,i[1]+2,'R'] in wrong_list_no_repetation:
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
        # print('between here')
        wrong_list_no_repetation=post_process_wrong_list(wrong_list_no_repetation,sent,new_sent)
        # print('between here 2')
        # wrong_list_no_repetation=[[str(i[0]+1),str(i[1]+1),i[2]]for i in wrong_list_no_repetation]
    return wrong_list_no_repetation
def get_final_text_edits(a,b)->list:
    # backtrackingPath2(a,b)
    jieba_word_list_process()
    spokenstr, writtenstr, operation, operation_list = backtrackingPath2(a, b)
    edit_list = post_process_operation_list(operation_list, a, b, print_flag=True)
    # edit_list=[[str(i[0]+1),str(i[1]+1),i[2]]for i in edit_list]
    return edit_list
def get_text_edits_on_token(a,b)->list:
    # backtrackingPath2(a,b)
    # jieba_word_list_process()
    try:
        spokenstr, writtenstr, operation, operation_list = backtrackingPath2(a, b)
        # print(len(operation_list),operation_list)
        # print(len(operation),operation)
        for i in operation:
            if len(i)==2:
                if not is_chinese(i[-1]):return []
            if len(i)==3:
                if is_chinese(i[1])==False or is_chinese(i[2])==False:return []
        # for index,
        edit_list = post_process_operation_list_on_token_level(operation_list, a, b, print_flag=False)
    except:
        return []
    # edit_list=[[str(i[0]+1),str(i[1]+1),i[2]]for i in edit_list]
    return edit_list
def get_jieba_cut_result(a):
    a_list=jieba.cut(a)
    # b_list=jieba.cut(b)
    a_jieba_cut='/'.join(a_list)
    # b_jieba_cut='/'.join(b_list)
    return a_jieba_cut
def jieba_word_list_process():
    wrong_dict_list=['真不知道','举子','这件','一件','来说','来看','更深','太夜','走白','喜喜','看过','挑早','的话','答到','人来','人去','盼了盼','听自',
                     '记对','环境保护','保护环境',]#需要删除的词语
    for i in wrong_dict_list:jieba.del_word(i)
    word_dict_list=['卖买','月尾','周所众知','彼起此伏']#需要添加的词语
    for i in wrong_dict_list:jieba.add_word(i)
if __name__ == '__main__':

    '''
    {"year": "2016", "text": "对我们国家来说，帮挨饿的人是当然做的事情。", 
    "correction": "对我们国家来说，帮挨饿的人是当然要做的事情。", 
    "wrong": [["17", "17", "M"]]}

    '''
    jieba_word_list_process()
    a='作物的产量又会大大降低。人民却没有东西吃'
    b='作物的生产量又会大大降低。人们却没有东西吃'
    a='我是谁'
    b='我谁是'
    a='对这个问题我的意见是更重要。'
    b='对这个问题我的意见是产生量更重要。'
    a='到底是健康重要，还是粮食生产量重要呢？对这个问题我的意见是更重要。'
    b='到底是健康重要，还是粮食生产量重要呢？对这个问题我的意见是产生量更重要。'
    a='重要？这个问题我的意见是更重要。'
    b='重要呢？对这个问题我的意见是产生量更重要。'
    # a='还有你你是一个家庭的爸爸'
    # b='还有，如果你是一个家庭的爸爸'
    a='倾心交谈你互相充分理解对方'
    b='倾心交谈，互相充分理解对方'
    wrong_list=get_text_edits_on_token(a,b)
    print(a)
    print(b)
    print('token wrong list',wrong_list)
    # spokenstr, writtenstr, operation,operation_list=backtrackingPath2(a,b)
    # wrong_list=post_process_operation_list_on_token_level(operation_list,a,b,print_flag=True)
    # print(a)
    # print(b)
    # print(wrong_list)

    #
    #
    # spokenstr, writtenstr, operation,operation_list=backtrackingPath2(a,b)
    # print(spokenstr)
    # print(writtenstr)
    # print(operation)
    # print(len(a),len(operation_list),operation_list)
    # wrong_list=post_process_operation_list(operation_list,a,b,print_flag=True)
    # print('wrong list',wrong_list)
    #
    # evaluation_flag=False
    # case_study_flag=True
    #
    # evaluation_flag=True
    # case_study_flag=False
    # if not evaluation_flag:
    #     pass
    # else:
    #     # total_data=read_json_fromTxt('../../data/CGED_Data/train_processed_processed.txt')
    #     total_data=read_json_fromTxt('./data/train_processed_processed.txt')
    #     # total_data=read_json_fromTxt('./data/train_processed.txt')
    #     total_gold_error=0.
    #     total_pred_error=0.
    #     total_pred_right_error=0.
    #
    #     total_gold_error_S=0.
    #     total_pred_error_S=0.
    #     total_pred_right_error_S=0.
    #
    #     total_gold_error_M=0.
    #     total_pred_error_M=0.
    #     total_pred_right_error_M=0.
    #
    #     total_gold_error_R=0.
    #     total_pred_error_R=0.
    #     total_pred_right_error_R=0.
    #
    #     total_gold_error_W=0.
    #     total_pred_error_W=0.
    #     total_pred_right_error_W=0.
    #
    #     total_wrong_num_for_each_sent=0
    #     for index,i in enumerate(total_data):
    #         if index%100==0:print(index)
    #         a=i['text']
    #         b=i['correction']
    #         if a==''or b=='':continue
    #         # print(index,a,b)
    #         spokenstr, writtenstr, operation, operation_list = backtrackingPath2(a, b)
    #         wrong_list=post_process_operation_list(operation_list,a,b)
    #         # wrong_list=[[str(i[0]+1),str(i[1]+1),i[2]]for i in wrong_list] #index先按照从0开始，计算
    #         gold_wrong_list=i['wrong']
    #         a_list = list(jieba.cut(a))
    #         b_list = list(jieba.cut(b))
    #         # if sorted(gold_wrong_list) != sorted(wrong_list) and len(a_list) == len(b_list):  # 对于词粒度个数相等，可以直接对齐的情况，debug
    #         if sorted(gold_wrong_list)!=sorted(wrong_list) and len(a_list)!=len(b_list):
    #             total_wrong_num_for_each_sent += 1
    #             if case_study_flag:
    #                 # diff_word_num,_=get_diff_word_num_and_list(a_list,b_list)
    #                 # if diff_word_num==1:
    #                 print(index,a,get_jieba_cut_result(a))
    #                 print(index,b,get_jieba_cut_result(b))
    #                 print('gold',gold_wrong_list)
    #                 print('pred',sorted(wrong_list))
    #
    #
    #         total_gold_error+=len(gold_wrong_list)
    #         total_pred_error+=len(wrong_list)
    #         each_pred_right_num=len([i for i in wrong_list if i in gold_wrong_list])
    #         total_pred_right_error+=each_pred_right_num
    #         ##计算各类型分别的p,r,f
    #         total_gold_error_S+=len([i for i in  gold_wrong_list if i[-1]=='S'])
    #         total_pred_error_S+=len([i for i in wrong_list if i[-1]=='S'])
    #         each_pred_right_num_S=len([i for i in wrong_list if (i in gold_wrong_list and i[-1]=='S')])
    #         total_pred_right_error_S+=each_pred_right_num_S
    #
    #         total_gold_error_M+=len([i for i in  gold_wrong_list if i[-1]=='M'])
    #         total_pred_error_M+=len([i for i in wrong_list if i[-1]=='M'])
    #         each_pred_right_num_M=len([i for i in wrong_list if (i in gold_wrong_list and i[-1]=='M')])
    #         total_pred_right_error_M+=each_pred_right_num_M
    #
    #         total_gold_error_R+=len([i for i in  gold_wrong_list if i[-1]=='R'])
    #         total_pred_error_R+=len([i for i in wrong_list if i[-1]=='R'])
    #         each_pred_right_num_R=len([i for i in wrong_list if (i in gold_wrong_list and i[-1]=='R')])
    #         total_pred_right_error_R+=each_pred_right_num_R
    #
    #         total_gold_error_W+=len([i for i in  gold_wrong_list if i[-1]=='W'])
    #         total_pred_error_W+=len([i for i in wrong_list if i[-1]=='W'])
    #         each_pred_right_num_W=len([i for i in wrong_list if (i in gold_wrong_list and i[-1]=='W')])
    #         total_pred_right_error_W+=each_pred_right_num_W
    #
    #
    #
    #     p=total_pred_right_error/(total_pred_error+0.001)
    #     r=total_pred_right_error/(total_gold_error+0.001)
    #     f=2*p*r/(p+r+0.001)
    #     print('total gold error',total_gold_error)
    #     print('total pred error',total_pred_error)
    #     print('total pred right error',total_pred_right_error)
    #     print('total wrong pred num for each sent',total_wrong_num_for_each_sent)
    #     print('p:',p)
    #     print('r:',r)
    #     print('f:',f)
    #
    #     p_S=total_pred_right_error_S/(total_pred_error_S+0.001)
    #     r_S=total_pred_right_error_S/(total_gold_error_S+0.001)
    #     f_S=2*p_S*r_S/(p_S+r_S+0.001)
    #     print('S total gold error',total_gold_error_S)
    #     print('S total pred error',total_pred_error_S)
    #     print('S total pred right error',total_pred_right_error_S)
    #     print('S p:',p_S)
    #     print('S r:',r_S)
    #     print('S f:',f_S)
    #
    #     p_M = total_pred_right_error_M / (total_pred_error_M + 0.001)
    #     r_M = total_pred_right_error_M / (total_gold_error_M + 0.001)
    #     f_M = 2 * p_M * r_M / (p_M + r_M + 0.001)
    #     print('M total gold error', total_gold_error_M)
    #     print('M total pred error', total_pred_error_M)
    #     print('M total pred right error', total_pred_right_error_M)
    #     print('M p:', p_M)
    #     print('M r:', r_M)
    #     print('M f:', f_M)
    #
    #     p_R = total_pred_right_error_R / (total_pred_error_R + 0.001)
    #     r_R = total_pred_right_error_R / (total_gold_error_R + 0.001)
    #     f_R = 2 * p_R * r_R / (p_R + r_R + 0.001)
    #     print('R total gold error', total_gold_error_R)
    #     print('R total pred error', total_pred_error_R)
    #     print('R total pred right error', total_pred_right_error_R)
    #     print('R p:', p_R)
    #     print('R r:', r_R)
    #     print('R f:', f_R)
    #
    #     p_W = total_pred_right_error_W / (total_pred_error_W + 0.001)
    #     r_W = total_pred_right_error_W / (total_gold_error_W + 0.001)
    #     f_W = 2 * p_W * r_W / (p_W + r_W + 0.001)
    #     print('W total gold error', total_gold_error_W)
    #     print('W total pred error', total_pred_error_W)
    #     print('W total pred right error', total_pred_right_error_W)
    #     print('W p:', p_W)
    #     print('W r:', r_W)
    #     print('W f:', f_W)
    #

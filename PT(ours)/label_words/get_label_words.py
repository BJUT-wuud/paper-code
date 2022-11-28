import jieba
import json
import os
import torch
from sentence_transformers import SentenceTransformer,util
import util as my_util


model_path = 'E://models//chinese-roberta-wwm-ext'
model = SentenceTransformer(model_path)

# dataset_name = 'tnews'
# dataset_name = 'cnews'
dataset_name = 'csldcp'


# cnews_params = 300, 6, 0.7  # cnews 超参数
# tnews_params = 40, 3, 0  # tnews超参数
csldcp_params = 100, 1, 0

datasets, all_core_words = my_util.get_csldcp_data()
top_k, FREQ, SCORE = csldcp_params


filter_words = []
filter_embeddings = []
filter_words_dict = []
# words_length = {}

# 筛选词频大于freq的词, 语义相似度不在这里筛选
def get_first_filter_words(freq, score):
    global filter_words
    global filter_embeddings
    if len(filter_words) > 0:
        return filter_words
    for idx, dataset in enumerate(datasets.items()):
        counts = {}
        after_dict = {}
        res = []
        print('=' * 100, idx, all_core_words[idx][0], '=' * 100)
        print(len(dataset[1]))
        # print(dataset[1][:100])
        # all_counts[dataset[0]] = {}
        for line in dataset[1]:
            my_util.seg_word(line, counts)

        # counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        # print(json.dumps(counts))
        core_words = all_core_words[idx]
        core_embeddings = my_util.trans_to_embedding(core_words)
        core_embedding_avg = my_util.get_embeddings_avg(core_embeddings)
        print('before filter : ', len(counts))
        word_embeddings = []
        # 拿到筛选后的词语
        for item in counts.items():
            word = item[0]
            word_freq = item[1]
            # print(item)
            # print(word_freq)
            if word_freq <= freq:
                continue

            word_embedding = model.encode(word, convert_to_tensor=True)
            # score_num = util.pytorch_cos_sim(core_embedding_avg, word_embedding)
            # if score_num > score:
            res.append(word)
            word_embeddings.append(word_embedding)
            after_dict[word] = word_freq
        filter_words.append(res)
        filter_embeddings.append(word_embeddings)
        filter_words_dict.append(after_dict)
        print('after filter: ', len(res))
        # words_length[map[dataset[0]]] = len(res)
        assert len(res) == len(word_embeddings)
    return filter_words, filter_embeddings

def filter_by_sim(core_embdding, word_embeddings):
    res = []
    for word_embedding in word_embeddings:
        sim = util.pytorch_cos_sim(core_embdding, word_embedding)
        if sim > SCORE:
            res.append(word_embedding)
    return res


# 开始之前先把所有的初次筛选的词放入全局list变量中
get_first_filter_words(FREQ, SCORE)
print(filter_words)
# print(words_length)


final_res = []
str_res = ''
for i in range(len(all_core_words)):
    print('=' * 50, all_core_words[i][0], '=' * 50)
    label_words = filter_words[i]
    words_embedding = filter_embeddings[i]
    core_words = all_core_words[i]
    words_dict = filter_words_dict[i]
    word_set = set()
    filter_dict = {}
    # sorted_final_words = []
    for core_word in core_words:
        core_word_embedding = model.encode(core_word, convert_to_tensor=True)
        # 在进行语义搜索前，对语义相似度小于SCORE的词进行过滤
        words_embedding = filter_by_sim(core_word_embedding, words_embedding)
        if len(words_embedding) == 0:
            continue
        # 进行语义搜索
        search_res = util.semantic_search(query_embeddings=core_word_embedding, corpus_embeddings=words_embedding,
                                          top_k=top_k)
        # 从search_res中拿到词的下标
        word_indexs = []
        final_words = []
        for word_index in search_res[0]:
            index = word_index['corpus_id']
            word_indexs.append(index)
            final_words.append(label_words[index])
        print(final_words)
        print('after final words: ', len(final_words))
        my_util.list_to_set(word_set, final_words)

    final_res.append(word_set)
    str_res += my_util.trans_list_to_string(word_set)
    str_res += '\n'


with open(f'res/{dataset_name}.txt', 'w', encoding='utf-8') as f:
    f.write(str_res)
    print(str_res)
    print('写入文件成功！')
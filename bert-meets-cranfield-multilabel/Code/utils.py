'''
    for BERT-BM25-Thesis-Project/bert-meets-cranfield-multilabel/Code/utils.py
'''
import numpy as np
import torch
from scipy import stats
import csv
from rank_bm25 import BM25Okapi
from operator import itemgetter
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
import random


def initialize_random_generators(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_gpu_device():
    if torch.cuda.is_available():
        print('GPU Type:', torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        return torch.device("cpu")


def reciprocal_rank(retrieved):
    retrieved = np.asarray(retrieved).nonzero()[0]
    return 1. / (retrieved[0] + 1) if retrieved.size else 0.


def precision_at_k(retrieved, k):
    assert k >= 1
    retrieved = np.asarray(retrieved)[:k] != 0
    if retrieved.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(retrieved)


def average_precision(retrieved, golds):
    retrieved = np.asarray(retrieved) != 0
    out = [precision_at_k(retrieved, k + 1) for k in range(retrieved.size) if retrieved[k]]
    if not out:
        return 0.
    return np.sum(out[:golds]) / golds


def dcg(retrieved, k):
    retrieved = np.asfarray(retrieved)[:k]
    if retrieved.size:
        return np.sum(retrieved / np.log2(np.arange(2, retrieved.size + 2)))


def ndcg(retrieved, gold_list, k):
    dcg_max = dcg(sorted(gold_list, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg(retrieved, k) / dcg_max


def retrieve_gold(rel_fed, query_num):
    res_list = [i for i in range(len(rel_fed)) if rel_fed[i][0] == query_num + 1]
    gold = [rel_fed[i][1:] for i in res_list]
    gold_unsorted = list(zip(*gold))[0]
    gold_sorted = sorted(gold, key=lambda l1: l1[1])
    gold_sorted_score = [5 - gs for gs in list(zip(*gold_sorted))[1]]
    return gold, gold_unsorted, gold_sorted_score


def get_binary_labels(rel_fed, multilabel=True):
    '''
        returns relevance labels, if multilabel=True the relevance grade will be returned,
        otherwise only 0 or 1 will be returned, indicating not relevant or relevant
    '''
    labels = []
    for query_num in range(0, 225):
        gold, gold_unsorted, gold_sorted_score = retrieve_gold(rel_fed, query_num)
        current_labels = np.zeros(1401)
        if multilabel:
            gold_unsorted_score = [l[1] for l in gold]
            current_labels[list(gold_unsorted)] = gold_unsorted_score
        else:
            current_labels[list(gold_unsorted)] = 1
        current_labels = current_labels[1:]
        labels.append(current_labels)
    return labels


def get_bm25_top_results(tokenized_corpus, tokenized_queries, n):
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_top_n = []
    query_num = 0

    for query in tokenized_queries:
        doc_scores = bm25.get_scores(query)
        feedback = list(zip(doc_scores, range(0, 1400)))
        feedback_sorted = sorted(feedback, reverse=True)
        bm25_top_n.append(list(zip(*feedback_sorted))[1][:n])
        query_num += 1
    return bm25, bm25_top_n


def bert_tokenizer(mode, bm25_top_n, corpus, labels, queries, max_length, model_type):
    padded, attention_mask, token_type_ids = [], [], []
    tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=True)
    temp_feedback = []

    for query_num in range(0, 225):
        if mode == 'Re-ranker':
            doc_nums = bm25_top_n[query_num]
            temp_corpus = list(itemgetter(*doc_nums)(corpus))
            temp_feedback.append(list(itemgetter(*doc_nums)(labels[query_num])))
        else:
            temp_corpus = corpus

        current_padded, current_attention_mask, current_token_type_ids = [], [], []
        for document in temp_corpus:
            current_encoded = tokenizer(queries[query_num],
                                        document,
                                        truncation='only_second',
                                        max_length=max_length,
                                        pad_to_max_length=True,
                                        return_tensors='pt')
            current_padded.append(current_encoded['input_ids'])
            current_attention_mask.append(current_encoded['attention_mask'])
            current_token_type_ids.append(current_encoded['token_type_ids'])

        padded.append(current_padded)
        attention_mask.append(current_attention_mask)
        token_type_ids.append(current_token_type_ids)
    return padded, attention_mask, token_type_ids, temp_feedback


def mrr_map_ndcg(trec_fold, current_query, query_num, feedback, rel_fed, mrr_total, map_total, ndcg_total, mrr_list,
                 map_list, ndcg_list, mode, fold_number, map_cut, ndcg_cut, multi_label_sel_cand_sorted=None,
                 multi_label_sel_cand_id=None, multilabel=False, current_label=-1):
    if current_label == 0:
        # label 0 indicates irrelevance, so the sorting should be the other way arround.
        feedback_sorted = sorted(feedback, reverse=False)
    else: 
        feedback_sorted = sorted(feedback, reverse=True)

    walker = 1
    for fs in feedback_sorted:
        current_fold = current_query
        current_fold += "D" + str(fs[1] + 1) + "\t" + str(walker) + "\t" + str(fs[0]) + "\t" + "run" + "\n"
        trec_fold += current_fold
        walker += 1

    text_file = open("../Output_Folder/result-" + mode + "-" + "Fold" + str(fold_number), "w")
    text_file.write(trec_fold)
    text_file.close()

    gold, gold_unsorted, gold_sorted_score = retrieve_gold(rel_fed, query_num)
    selected_candidates_id = list(zip(*feedback_sorted))[1]
    selected_candidates = [1 if s + 1 in gold_unsorted else 0 for s in selected_candidates_id]
    selected_candidates_sorted = [(5 - gold[gold_unsorted.index(s + 1)][1]) if (s + 1) in gold_unsorted else 0 for s in
                                  selected_candidates_id[:20]]

    if multilabel and multi_label_sel_cand_id is not None:
        selected_candidates_sorted = multi_label_sel_cand_sorted
        selected_candidates_id = multi_label_sel_cand_id
        selected_candidates = [1 if s + 1 in gold_unsorted else 0 for s in selected_candidates_id]

    mrr_current = reciprocal_rank(selected_candidates[:map_cut])
    mrr_list.append(mrr_current)
    mrr_total += mrr_current

    map_current = average_precision(selected_candidates[:map_cut], len(gold_unsorted))
    map_list.append(map_current)
    map_total += map_current

    ndcg_current = ndcg(selected_candidates_sorted, gold_sorted_score, ndcg_cut)
    ndcg_list.append(ndcg_current)
    ndcg_total += ndcg_current

    return mrr_total, map_total, ndcg_total, mrr_list, map_list, ndcg_list, trec_fold


def get_bm25_results(mrr_bm25_list, map_bm25_list, ndcg_bm25_list, test_index, tokenized_queries, bm25, mrr_bm25,
                     map_bm25, ndcg_bm25, rel_fed, fold_number, map_cut, ndcg_cut):
    mrr_total, map_total, ndcg_total = 0, 0, 0
    mrr_list, map_list, ndcg_list = [], [], []
    trec_fold = ""
    for query_index in test_index:
        current_query = "Q" + str(query_index + 1) + "\t" + "Q0" + "\t"
        query = tokenized_queries[query_index]
        doc_scores = bm25.get_scores(query)
        feedback = list(zip(doc_scores, range(0, 1400)))
        mrr_total, map_total, ndcg_total, mrr_list, map_list, ndcg_list, trec_fold = mrr_map_ndcg(trec_fold,
                                                                                                  current_query,
                                                                                                  query_index, feedback,
                                                                                                  rel_fed, mrr_total,
                                                                                                  map_total, ndcg_total,
                                                                                                  mrr_list, map_list,
                                                                                                  ndcg_list, 'BM25',
                                                                                                  fold_number,
                                                                                                  map_cut, ndcg_cut)

    mrr_bm25_list += list(zip(mrr_list, test_index))
    map_bm25_list += list(zip(map_list, test_index))
    ndcg_bm25_list += list(zip(ndcg_list, test_index))

    mrr_bm25 += mrr_total / len(test_index)
    map_bm25 += map_total / len(test_index)
    ndcg_bm25 += ndcg_total / len(test_index)
    print("MRR:  " + "{:.4f}".format(mrr_total / len(test_index)))
    print("MAP:  " + "{:.4f}".format(map_total / len(test_index)))
    print("NDCG: " + "{:.4f}".format(ndcg_total / len(test_index)))
    print(len(map_bm25_list))
    return mrr_bm25, map_bm25, ndcg_bm25, mrr_bm25_list, map_bm25_list, ndcg_bm25_list


def model_preparation(MODEL_TYPE, train_dataset, test_dataset, batch_size, batch_size_test, learning_rate, epochs, model=None, num_labels=2):
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=batch_size)

    test_dataloader = DataLoader(test_dataset,
                                 sampler=None,
                                 batch_size=batch_size_test)

    if model is None:
        model = BertForSequenceClassification.from_pretrained(
            MODEL_TYPE,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
    else:
        model = model
    torch.cuda.empty_cache()
    model.cuda()

    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=1e-8)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    return train_dataloader, test_dataloader, model, optimizer, scheduler


def training(model, train_dataloader, device, optimizer, scheduler):
    total_train_loss = []
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 100 == 0 and step != 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_input_token = batch[2].to(device)
        b_labels = batch[3].to(device)

        model.zero_grad()
        outputs = model(b_input_ids,
                             token_type_ids=b_input_token,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits
        total_train_loss.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = sum(total_train_loss) / len(train_dataloader)
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    return model, optimizer, scheduler


def testing(mode, model, test_dataloader, device, test_index, bm25_top_n, mrr_bert_list, map_bert_list, ndcg_bert_list,
            mrr_bert, map_bert, ndcg_bert, rel_fed, fold_number, map_cut, ndcg_cut, multilabel=False, argmax_sorting=False,
            per_label_testing=False):
    model.eval()
    predictions, true_labels = [], []
    walker = 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_input_token, b_labels = batch

        outputs_list = []
        with torch.no_grad():
            if len(b_input_ids[0]) > 100:
                for i in range(0, 14):
                    torch.cuda.empty_cache()
                    current_b_input_ids = b_input_ids[100 * i:100 * (i + 1)]
                    current_b_input_mask = b_input_mask[100 * i:100 * (i + 1)]
                    current_b_input_token = b_input_token[100 * i:100 * (i + 1)]
                    outputs_list.append(model(current_b_input_ids, token_type_ids=current_b_input_token,
                                              attention_mask=current_b_input_mask))
            else:
                outputs = model(b_input_ids, token_type_ids=b_input_token, attention_mask=b_input_mask)
        logits = []
        if len(outputs_list):
            # logits = torch.cat(outputs_list, dim=0)
            for i in range(len(outputs_list)):
                if len(logits) > 0:
                    logits = np.append(logits, outputs_list[i][0].detach().cpu().numpy(), axis=0)
                else:
                    logits = outputs_list[i][0].detach().cpu().numpy()
            #    logits = np. outputs_list[i][0].detach().cpu().numpy()
        else:
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)
        walker += 1

    if per_label_testing:
        begin_range = 0
        end_range   = 5
    else:
        begin_range = 1
        end_range   = 1
    for label_index in range(begin_range, end_range):
        walker = 0
        mrr_total, map_total, ndcg_total = 0, 0, 0
        mrr_list, map_list, ndcg_list = [], [], []
        trec_fold = ""
        if per_label_testing:
            print('===================================')
            print('PER_LABEL_TESTING: LABEL ', label_index)
            print('===================================')
        for query_index in test_index:
            current_query = "Q" + str(query_index + 1) + "\t" + "Q0" + "\t"
            doc_scores = list(zip(*predictions[walker]))[label_index]

            # doc_scores = list(zip(*predictions[walker]))[1]
            if mode == 'Re-ranker':
                feedback = list(zip(doc_scores, bm25_top_n[query_index]))
            else:
                feedback = list(zip(doc_scores, range(0, 1400)))

            ## do arg_max sorting
            if argmax_sorting:
                # note that feedback is not changed, because it is not used any further in this case
                # apply softmax on scores, this way, we can invert the predictions for label 0, they are the other way around
                current_predictions = np.asarray(predictions[walker])
                category_score = np.max(current_predictions, axis=1)  # score for category
                category = np.argmax(current_predictions, axis=1)  # predicted category

                # t_current_predictions = torch.tensor(current_predictions)
                # m = torch.nn.Softmax(dim=1)
                # t_softmax_current_predictions = m(t_current_predictions)
                # # back to numpy, for easier handling
                # np_current_predictions = np.asarray(t_cur_pre_probs)
                # # invert cat 0
                # category = np.argmax(current_predictions, axis=1)  # predicted category
                # category_score = np.max(np_current_predictions, axis=1)  # score for category

                category_rev = np.where(category > 0, 5 - category, 0)
                # make structured array because it enables multivariate sorting (ascending only)
                m_selected_candidates = np.core.records.fromarrays([bm25_top_n[query_index], category_rev, category_score],
                                                                   names='bm25_top_n_qi, category, category_score')
                # m_selected_candidates_sorted = np.sort(m_selected_candidates, order=['category', 'category_score'])

                # TEST split for different sorting
                m_selected_candidates_notrel = m_selected_candidates[m_selected_candidates['category'] == 0]
                m_selected_candidates_rel    = m_selected_candidates[m_selected_candidates['category'] > 0]
                m_selected_candidates_not_rel_sorted = np.sorted(m_selected_candidates_rel, order=['category_score']) 
                m_selected_candidates_rel_sorted = np.sort(m_selected_candidates_rel, order=['category', 'category_score'])
                m_selected_candidates_rel_sorted = np.flip(m_selected_candidates_rel_sorted)
                m_selected_candidates_sorted = np.hstack(m_selected_candidates_rel_sorted, m_selected_candidates_not_rel_sorted)

                # the sorting is ascending, flipping it makes the order right
                # m_selected_candidates_sorted = np.flip(m_selected_candidates_sorted)
                ####### END TEST

                # set the candidate ids
                # m_selected_candidates_id = m_selected_candidates_sorted['bm25_top_n_qi'][:20]
                m_selected_candidates_id = m_selected_candidates_sorted['bm25_top_n_qi']
                gold, gold_unsorted, gold_sorted_score = retrieve_gold(rel_fed, query_index)
                # m_selected_candidates = [1 if s + 1 in gold_unsorted else 0 for s in m_selected_candidates_id] #oboslete
                m_selected_candidates_sorted = [(5 - gold[gold_unsorted.index(s + 1)][1]) if (s + 1) in gold_unsorted else 0 for s in
                                  m_selected_candidates_id[:20]]
            else:
                m_selected_candidates_sorted = None
                m_selected_candidates_id = None
            mrr_total, map_total, ndcg_total, mrr_list, map_list, ndcg_list, trec_fold = mrr_map_ndcg(trec_fold,
                                                                                                      current_query,
                                                                                                      query_index, feedback,
                                                                                                      rel_fed, mrr_total,
                                                                                                      map_total, ndcg_total,
                                                                                                      mrr_list, map_list,
                                                                                                      ndcg_list, 'BERT',
                                                                                                      fold_number, map_cut,
                                                                                                      ndcg_cut,
                                                                                                      multi_label_sel_cand_sorted=m_selected_candidates_sorted,
                                                                                                      multi_label_sel_cand_id=m_selected_candidates_id,
                                                                                                      multilabel=multilabel,
                                                                                                      current_label=label_index)
            walker += 1

        mrr_bert_list += list(zip(mrr_list, test_index))
        map_bert_list += list(zip(map_list, test_index))
        ndcg_bert_list += list(zip(ndcg_list, test_index))

        mrr_bert += mrr_total / len(test_index)
        map_bert += map_total / len(test_index)
        ndcg_bert += ndcg_total / len(test_index)
        print("  Test MRR:  " + "{:.4f}".format(mrr_total / len(test_index)))
        print("  Test MAP:  " + "{:.4f}".format(map_total / len(test_index)))
        print("  Test NDCG: " + "{:.4f}".format(ndcg_total / len(test_index)))
        print(len(map_bert_list))
    return mrr_bert, map_bert, ndcg_bert, mrr_bert_list, map_bert_list, ndcg_bert_list


def t_test(bm25_list, bert_list, measure):
    prediction_bm25 = list(zip(*bm25_list))[0]
    prediction_bert = list(zip(*bert_list))[0]
    t_value, p_value = stats.ttest_ind(prediction_bm25, prediction_bert)
    # print("t-value " + measure + ": " + "{:.4f}".format(t_value))
    print("p-value " + measure + ": " + "{:.4f}".format(p_value))


def results_to_csv(location, result_list):
    with open(location, "w") as f:
        writer = csv.writer(f)
        writer.writerows(result_list)

        
# ADDITIONAL utils
def get_bm25_plus_other_rel(bm25_tn, labels, queries):
      bm25_top_n_rel_padded = [0]*len(queries) # a bm25_top_n list padded with the remaining relevant documents
      bm25_top_n_swap = [0]*len(queries) 
    
      for qi in range(len(queries)):
        # get the list of relelvant documents
        lbi = np.where(labels[qi] == 1)
        # note this numbering is only compatible with the labels list


        # get the list of bm25_top_n
#         np_bm25_qi_docs = np.array(bm25_top_n[qi]) 
        np_bm25_qi_docs = np.array(bm25_tn[qi]) 

        # evaluate what relevant documents should be added
        pad_rel = np.setdiff1d(lbi, np_bm25_qi_docs)
        # if len(pad_rel) > 0:
        pad_rel = tuple(pad_rel)
#         bm25_top_n_rel_padded[qi] = bm25_top_n[qi] + pad_rel
        bm25_top_n_rel_padded[qi] = bm25_tn[qi] + pad_rel
        # create a list with least relevant items swapped for unfound relevant
        for i in range(len(pad_rel)):
          # CHECK
          # are we to swap a relevant document?
          current_doc = np_bm25_qi_docs[-(i+1)] 
          
          if np.count_nonzero(current_doc == lbi) > 0:
            print('Relevant doc overwritten!')
          # CONTINUE  
          np_bm25_qi_docs[-(i+1)] = pad_rel[i]
          
        bm25_top_n_swap[qi] = np_bm25_qi_docs
      return bm25_top_n_rel_padded, bm25_top_n_swap
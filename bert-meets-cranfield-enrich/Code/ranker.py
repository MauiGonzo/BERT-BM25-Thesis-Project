import utils
import data_utils
from operator import itemgetter
import os

# ========================================
#               Hyper-Parameters
# ========================================
SEED = 76
MODE = 'Full-ranker'
MODEL_TYPE = 'bert-base-uncased'
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 2
TOP_BM25 = 100
MAP_CUT = 100
NDCG_CUT = 20
if MODE == 'Full-ranker':
    TEST_BATCH_SIZE = 1400
else:
    TEST_BATCH_SIZE = 100

# Set the seed value all over the place to make this reproducible.
utils.initialize_random_generators(SEED)

if __name__ == "__main__":
    print("# ========================================")
    print("#               Hyper-Parameters")
    print(MODE)
    print(MODEL_TYPE)
    print(LEARNING_RATE)
    print(MAX_LENGTH)
    print(BATCH_SIZE)
    print(EPOCHS)
    print("# ========================================")

    device = utils.get_gpu_device()
    if not os.path.exists('../Output_Folder'):
        os.makedirs('../Output_Folder')

    queries = data_utils.get_queries('../Data/cran/cran.qry')
    corpus = data_utils.get_corpus('../Data/cran/cran.all.1400')
    rel_fed = data_utils.get_judgments('../Data/cran/cranqrel')

    labels = utils.get_binary_labels(rel_fed)
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    tokenized_queries = [query.split(" ") for query in queries]

    bm25, bm25_top_n = utils.get_bm25_top_results(tokenized_corpus, tokenized_queries, TOP_BM25)

    padded_all, attention_mask_all, token_type_ids_all, temp_feedback = utils.bert_tokenizer(MODE, bm25_top_n, corpus,
                                                                                             labels, queries,
                                                                                             MAX_LENGTH, MODEL_TYPE)

    # ========================================
    #               Folds
    # ========================================
    mrr_bm25_list, map_bm25_list, ndcg_bm25_list = [], [], []
    mrr_bert_list, map_bert_list, ndcg_bert_list = [], [], []
    mrr_bm25, map_bm25, ndcg_bm25 = 0, 0, 0
    mrr_bert, map_bert, ndcg_bert = 0, 0, 0

    for fold_number in range(1, 6):
        print('======== Fold {:} / {:} ========'.format(fold_number, 5))
        train_index, test_index = data_utils.load_fold(fold_number)

        padded, attention_mask, token_type_ids = [], [], []
        if MODE == 'Re-ranker':
            padded, attention_mask, token_type_ids = padded_all, attention_mask_all, token_type_ids_all
        else:
            temp_feedback = []
            for query_num in range(0, len(bm25_top_n)):
                if query_num in test_index:
                    doc_nums = range(0, 1400)
                else:
                    doc_nums = bm25_top_n[query_num]
                padded.append(list(itemgetter(*doc_nums)(padded_all[query_num])))
                attention_mask.append(list(itemgetter(*doc_nums)(attention_mask_all[query_num])))
                token_type_ids.append(list(itemgetter(*doc_nums)(token_type_ids_all[query_num])))
                temp_feedback.append(list(itemgetter(*doc_nums)(labels[query_num])))

        train_dataset = data_utils.get_tensor_dataset(train_index, padded, attention_mask, token_type_ids,
                                                      temp_feedback)
        test_dataset = data_utils.get_tensor_dataset(test_index, padded, attention_mask, token_type_ids, temp_feedback)

        mrr_bm25, map_bm25, ndcg_bm25, mrr_bm25_list, map_bm25_list, ndcg_bm25_list = utils.get_bm25_results(
            mrr_bm25_list, map_bm25_list, ndcg_bm25_list, test_index, tokenized_queries, bm25, mrr_bm25, map_bm25,
            ndcg_bm25, rel_fed, fold_number, MAP_CUT, NDCG_CUT)

        train_dataloader, test_dataloader, model, optimizer, scheduler = utils.model_preparation(MODEL_TYPE, train_dataset,
                                                                                                 test_dataset,
                                                                                                 BATCH_SIZE, TEST_BATCH_SIZE,
                                                                                                 LEARNING_RATE, EPOCHS)
        # ========================================
        #               Training Loop
        # ========================================
        epochs_train_loss, epochs_val_loss = [], []
        for epoch_i in range(0, EPOCHS):
            # ========================================
            #               Training
            # ========================================
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
            print('Training...')
            model, optimizer, scheduler = utils.training(model, train_dataloader, device, optimizer, scheduler)
        # ========================================
        #               Testing
        # ========================================
        print('Testing...')
        mrr_bert, map_bert, ndcg_bert, mrr_bert_list, map_bert_list, ndcg_bert_list = utils.testing(MODE, model,
                                                                                                    test_dataloader,
                                                                                                    device, test_index,
                                                                                                    bm25_top_n,
                                                                                                    mrr_bert_list,
                                                                                                    map_bert_list,
                                                                                                    ndcg_bert_list,
                                                                                                    mrr_bert, map_bert,
                                                                                                    ndcg_bert, rel_fed,
                                                                                                    fold_number,
                                                                                                    MAP_CUT, NDCG_CUT)
    print("  BM25 MRR:  " + "{:.4f}".format(mrr_bm25 / 5))
    print("  BM25 MAP:  " + "{:.4f}".format(map_bm25 / 5))
    print("  BM25 NDCG: " + "{:.4f}".format(ndcg_bm25 / 5))

    print("  BERT MRR:  " + "{:.4f}".format(mrr_bert / 5))
    print("  BERT MAP:  " + "{:.4f}".format(map_bert / 5))
    print("  BERT NDCG: " + "{:.4f}".format(ndcg_bert / 5))

    utils.t_test(mrr_bm25_list, mrr_bert_list, 'MRR')
    utils.t_test(map_bm25_list, map_bert_list, 'MAP')
    utils.t_test(ndcg_bm25_list, ndcg_bert_list, 'NDCG')

    # utils.results_to_csv('./mrr_bm25_list.csv', mrr_bm25_list)
    # utils.results_to_csv('./mrr_bert_list.csv', mrr_bert_list)
    # utils.results_to_csv('./map_bm25_list.csv', map_bm25_list)
    # utils.results_to_csv('./map_bert_list.csv', map_bert_list)
    # utils.results_to_csv('./ndcg_bm25_list.csv', ndcg_bm25_list)
    # utils.results_to_csv('./ndcg_bert_list.csv', ndcg_bert_list)

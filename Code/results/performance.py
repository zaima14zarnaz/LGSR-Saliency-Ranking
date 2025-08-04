from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from statistics import median
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
import pandas as pd
import numpy as np

class Performance:
    @staticmethod
    def generate_ordering(lst):
        # sort unique elements and assign ranks
        sorted_vals = sorted(lst)
        val_to_rank = {val: rank + 1 for rank, val in enumerate(sorted_vals)}
        # map the original list to its ranks
        return [val_to_rank[x] for x in lst]

    @staticmethod
    def compute_fm_at_k(pred_labels, gt_labels, k):
        pred_topk = set(pred_labels[:k])
        gt_topk = set(gt_labels)
        tp = len(pred_topk & gt_topk)
        precision = tp / len(pred_topk) if pred_topk else 0
        recall = tp / len(gt_topk) if gt_topk else 0
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def results(rank_set1, rank_set2):
        spearman_scores = []
        pearson_scores = []
        mae_scores = []
        kendall_scores = []
        ndcg_scores = []

        total_mae = 0.0
        mae_count = 0

        # print(len(rank_set1))
        # print(len(rank_set2))

        for img in rank_set1:
            if img in rank_set2:
                
                rank_anns1 = rank_set1[img]
                rank_anns2 = rank_set2[img]

                max_mae = 0.0
                max_spearman = 0.0
                max_pearson = 0.0
                max_kendall = 0.0
                max_ndcg = 0.0
                valid_found = False
                for labet_set1 in rank_anns1:
                    # get common labels
                    common_labels = list(set(labet_set1) & set(rank_anns2))
                    
                    # assign ranks based on positions
                    rankings1 = [labet_set1.index(label) + 1 for label in common_labels]
                    rankings2 = [rank_anns2.index(label) + 1 for label in common_labels]


                    rankings1 = Performance.generate_ordering(rankings1)
                    rankings2 = Performance.generate_ordering(rankings2)
                    # rankings2.reverse()
                    

                    if len(rankings1) == len(rankings2) and len(rankings1) > 0:
                        mae = sum(abs(h - l) for h, l in zip(rankings1, rankings2)) / len(rankings1)
                        if mae > max_mae:
                            max_mae = mae

                    
                    # calculate correlations
                    if len(rankings1) >= 2:  # need at least 2 to correlate
                        valid_found = True
                        spearman_corr, _ = spearmanr(rankings1, rankings2)
                        pearson_corr, _ = pearsonr(rankings1, rankings2)
                        spearman_corr = round(spearman_corr, 1)
                        pearson_corr = round(pearson_corr, 1)
                        if pearson_corr > max_pearson:
                            max_pearson = pearson_corr
                        if spearman_corr > max_spearman:
                            max_spearman = spearman_corr
                        
                        tau_corr, _ = kendalltau(rankings1, rankings2)
                        if tau_corr > max_kendall:
                            max_kendall = tau_corr

                        # Prepare NDCG inputs
                        true_relevance = np.array([len(rankings1) - r for r in rankings1]).reshape(1, -1)
                        scores = np.array([len(rankings2) - r for r in rankings2]).reshape(1, -1)
                        ndcg = ndcg_score(true_relevance, scores)
                        if ndcg > max_ndcg:
                            max_ndcg = ndcg
                        
                        # if spearman_corr <= 0.8:
                        #     l1 = [label for label in labet_set1 if label in common_labels]
                        #     l2 = [label for label in rank_anns2 if label in common_labels]
                        #     print(img)
                        #     print(f'LLM_ranks: {l2}')
                        #     print(f'Human Rank: {l1}')
                        #     print(f'Spearman: {spearman_corr}')

                
                if valid_found:
                    total_mae += max_mae
                    mae_count += 1


                    spearman_scores.append(max_spearman)
                    pearson_scores.append(max_pearson)
                    kendall_scores.append(max_kendall)
                    ndcg_scores.append(max_ndcg)

        if len(spearman_scores) <= 0:
            return -1, -1, -1, -1, -1
        mean_mae = total_mae / mae_count if mae_count > 0 else 0
        # print(spearman_scores)
        # print(f'Sum of spearman: {sum(spearman_scores)} number: {len(spearman_scores)}')
        mean_spearman = sum(spearman_scores) / len(spearman_scores)
        mean_pearson = sum(pearson_scores) / len(pearson_scores)
        mean_kendall = sum(kendall_scores) / len(kendall_scores) if kendall_scores else 0
        mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0

        return round(mean_spearman,4), round(mean_pearson,4), round(mean_mae,4), round(mean_kendall, 4), round(mean_ndcg, 4)
    
    @staticmethod
    def compute_quartile_differences(df1, df2, df3):
        diff_1_2 = df2[['Q1', 'Median', 'Q3']] - df1[['Q1', 'Median', 'Q3']]
        diff_2_3 = df3[['Q1', 'Median', 'Q3']] - df2[['Q1', 'Median', 'Q3']]

        # Rename columns to indicate which diff
        diff_1_2.columns = pd.MultiIndex.from_product([['Diff_1_2'], diff_1_2.columns])
        diff_2_3.columns = pd.MultiIndex.from_product([['Diff_2_3'], diff_2_3.columns])

        # Concatenate horizontally
        result = pd.concat([diff_1_2, diff_2_3], axis=1)
        return result
    
    @staticmethod
    def spearman_for_string_lists(list1, list2, missing_rank='end'):
        # Step 1: Union of all unique items
        all_items = set(list1).union(set(list2))
        
        # Step 2: Create ranking dictionaries
        def get_ranks(lst, all_items, missing_rank):
            rank_dict = {item: rank for rank, item in enumerate(lst)}
            default_rank = len(lst) if missing_rank == 'end' else -1
            return [rank_dict.get(item, default_rank) for item in all_items]
        
        ranks1 = get_ranks(list1, all_items, missing_rank)
        ranks2 = get_ranks(list2, all_items, missing_rank)

        # Step 3: Compute Spearman correlation
        score, _ = spearmanr(ranks1, ranks2)
        return score
    

# region_complexity_path = {
#     'human': '/home/zaimaz/Desktop/research1/SaliencyRanking/Code/groundTruth/human_annotations/data_analysis_results/human/rgn_cmpx_vgg_quartles_Rank_human.csv',
#     'llm': '/home/zaimaz/Desktop/research1/SaliencyRanking/Code/groundTruth/human_annotations/data_analysis_results/llm/rgn_cmpx_vgg_quartles_Rank_llm.csv',
#     'assr': '/home/zaimaz/Desktop/research1/SaliencyRanking/Code/groundTruth/human_annotations/data_analysis_results/assr/rgn_cmpx_vgg_quartles_Rank_assr.csv',
#     'irsr': '/home/zaimaz/Desktop/research1/SaliencyRanking/Code/groundTruth/human_annotations/data_analysis_results/irsr/rgn_cmpx_vgg_quartles_Rank_irsr.csv',
#     'sifr': '/home/zaimaz/Desktop/research1/SaliencyRanking/Code/groundTruth/human_annotations/data_analysis_results/sifr/rgn_cmpx_vgg_quartles_Rank_sifr.csv'
# }


# diff = Performance.compute_quartile_differences(pd.read_csv(region_complexity_path['llm']),
#                                                 pd.read_csv(region_complexity_path['human']),
#                                                 pd.read_csv(region_complexity_path['irsr']))
# print(diff.head)





import torch
import time
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score, average_precision_score


# def mapk(actuals, predicted, k=0):
#   return np.mean([apk(a,p,k) for a,p in product([actuals], [predicted])])

def ndcg(y_true, y_pred, top_K=0):
    # From 趙儀
    ndcg_K = []
    true_sum = 0

    for i in range(y_pred.shape[0]):
        top_indices = y_pred[i].argsort()[-top_K:][::-1] #每篇文章排名前Top K可能的tag index
        true_num = np.sum(y_true[i, :])
        true_sum += true_num
        dcg = 0
        idcg = 0
        idcgCount = true_num
        j = 0
        for item in top_indices:
            if y_true[i, item] == 1:
                dcg += 1.0/math.log2(j + 2)
            if idcgCount > 0:
                idcg += 1.0/math.log2(j + 2)
                idcgCount = idcgCount-1
            j += 1
        if(idcg != 0):
            ndcg_K.append(dcg/idcg)

    return  np.mean(np.array(ndcg_K))

def test_model(args, test_loader, user_network, item_network, co_attention, fc_layer):
    # ---------- Test ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    user_network.eval()
    item_network.eval()
    co_attention.eval()
    fc_layer.eval()

    # These are used to record information in test.
    test_accs = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    # Iterate the test set by batches.
    print("-------------------------- TEST --------------------------")
    for batch in tqdm(test_loader):

        # We don't need gradient in test.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():

            # Exacute models 
            user_review_emb, item_review_emb, user_review_mask, item_review_mask, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
            user_logits = user_network(user_review_emb.to(args["device"]), user_review_mask.to(args["device"]), user_lda_groups.to(args["device"]))
            item_logits = item_network(item_review_emb.to(args["device"]), item_review_mask.to(args["device"]), item_lda_groups.to(args["device"]))
            weighted_user_logits,  weighted_item_logits = co_attention(user_logits, item_logits)
            user_feature = torch.cat((weighted_user_logits, user_mf_emb.to(args["device"])), dim=1)
            item_feature = torch.cat((weighted_item_logits, item_mf_emb.to(args["device"])), dim=1)
            fc_input = torch.cat((user_feature, item_feature), dim=1)
            output_logits = fc_layer(fc_input)

            # Output after sigmoid is greater than "Q" will be considered as 1, else 0.
            result_logits = torch.where(output_logits > 0.5, 1, 0).squeeze(dim=-1)
            labels = labels.to(args["device"])

            # Compute the information for current batch.
            acc = (result_logits == labels).float().mean()
            precision = precision_score(labels.cpu(), result_logits.cpu(), zero_division=0)
            recall = recall_score(labels.cpu(), result_logits.cpu())
            f1 = f1_score(labels.cpu(), result_logits.cpu())
            # ndcg = ndcg_score(labels.unsqueeze(dim=-1).cpu(), result_logits.unsqueeze(dim=-1).cpu())

            # Record the information.
            test_accs.append(acc)
            test_precisions.append(precision)
            test_recalls.append(recall)
            test_f1s.append(f1)
    

    # The average loss and accuracy for entire test set is the average of the recorded values.
    test_acc = sum(test_accs) / len(test_accs)
    test_precision = sum(test_precisions) / len(test_precisions)
    test_recall = sum(test_recalls) / len(test_recalls)
    test_f1 = sum(test_f1s) / len(test_f1s)

    print(f"[ Test base ] acc = {test_acc:.4f}, precision = {test_precision:.4f}, recall = {test_recall:.4f}, f1 = {test_f1:.4f}")
    with open('output/history/test_base.csv','a') as file:
        file.write(time.strftime("%m-%d %H:%M")+","+f"test,{test_acc:.4f},{test_precision:.4f},{test_recall:.4f},{test_f1:.4f}" + "\n")

def test_model_topk(args, test_loader, user_network, item_network, co_attention, fc_layer):
    # ---------- Test ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    user_network.eval()
    item_network.eval()
    co_attention.eval()
    fc_layer.eval()

    # To store the prediction
    predict_incidence_df = test_loader.dataset.get_empty_incidence_df()
    label_incidence_df = test_loader.dataset.get_true_incidence_df()
    
    # Iterate the test set by batches.
    print("-------------------------- TEST --------------------------")
    for batch in tqdm(test_loader):
        # We don't need gradient in test.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():

            # Exacute models 
            userId, itemId, user_review_emb, item_review_emb, user_review_mask, item_review_mask, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
            user_logits = user_network(user_review_emb.to(args["device"]), user_review_mask.to(args["device"]), user_lda_groups.to(args["device"]))
            item_logits = item_network(item_review_emb.to(args["device"]), item_review_mask.to(args["device"]), item_lda_groups.to(args["device"]))
            weighted_user_logits,  weighted_item_logits = co_attention(user_logits, item_logits)
            user_feature = torch.cat((weighted_user_logits, user_mf_emb.to(args["device"])), dim=1)
            item_feature = torch.cat((weighted_item_logits, item_mf_emb.to(args["device"])), dim=1)
            fc_input = torch.cat((user_feature, item_feature), dim=1)
            output_logits = fc_layer(fc_input)

            for user, item, logit in zip(userId.cpu(), itemId.cpu(), output_logits.squeeze(dim=-1).cpu()):
                predict_incidence_df.at[int(user), int(item)] = float(logit)

            # Output after sigmoid is greater than "Q" will be considered as 1, else 0.
            result_logits = torch.where(output_logits > 0.5, 1, 0).squeeze(dim=-1)
            labels = labels.to(args["device"])

    # For topk score calculation
    TOP_N = 10
    top_k_df = predict_incidence_df.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=TOP_N)

    # Save result
    predict_incidence_df.to_csv('output/history/probability_df.csv')
    predict_incidence_df.to_pickle('output/history/probability_df.pkl')
    top_k_df.to_csv('output/history/topk_prediction_df.csv')
    top_k_df.to_pickle('output/history/topk_prediction_df.pkl')
    print(predict_incidence_df)
    print(top_k_df)

    global_hit_10 = 0
    global_hit_5 = 0
    global_like = 0

    new_hit_5 = []
    new_hit_10 = []

    global_labels = []
    global_prediction = []

    # Calculate each score
    for user in predict_incidence_df.index:
        
        # Data for scoring
        user_topk_prediction = top_k_df.loc[user]
        user_pred_prob = predict_incidence_df.loc[user]
        
        user_label = label_incidence_df.loc[user]
        user_like = user_label[user_label==1].keys()
        user_topk_label = label_incidence_df.loc[user, user_topk_prediction]

        # get 01 prediction from probability
        pred_tensor = torch.tensor(user_pred_prob.values)
        value, idx = pred_tensor.topk(k=TOP_N)
        user_pred_binary = torch.zeros(pred_tensor.size())
        user_pred_binary[idx] = 1 

        global_like += len(user_like)
        global_labels.append(user_label.tolist())
        global_prediction.append(user_pred_binary.tolist())

        # Top-10 hit ratio
        user_top_10_label = user_topk_label[:10]
        global_hit_10 += len(user_top_10_label[user_top_10_label==1])
        if len(user_top_10_label[user_top_10_label==1]) > 0:
            new_hit_10.append(1)
        else:
            new_hit_10.append(0)

        # Top-5 hit ratio
        user_top_5_label = user_topk_label[:5]
        global_hit_5 += len(user_top_5_label[user_top_5_label==1])
        if len(user_top_5_label[user_top_5_label==1]) > 0:
            new_hit_5.append(1)
        else:
            new_hit_5.append(0)

    test_precision = precision_score(global_labels, global_prediction, zero_division=0, average="samples")
    test_recall = recall_score(global_labels, global_prediction, zero_division=0, average="samples")
    test_f1 = f1_score(global_labels, global_prediction, zero_division=0, average="samples")

    test_map = average_precision_score(label_incidence_df.to_numpy(), predict_incidence_df.to_numpy(), average="samples")
    test_ndcg = ndcg(label_incidence_df.to_numpy(), predict_incidence_df.to_numpy(), TOP_N)

    # test_top_10_hr = global_hit_10 / global_like
    # test_top_5_hr = global_hit_5 / global_like

    new_test_hit_10 = sum(new_hit_10) / len(new_hit_10)
    new_test_hit_5 = sum(new_hit_5) / len(new_hit_5)

    print(f"[ Test base ] precision@{TOP_N} = {test_precision:.4f}, recall@{TOP_N} = {test_recall:.4f}, f1@{TOP_N} = {test_f1:.4f}")
    print(f"[ Test base ] MAP@{TOP_N} = {test_map:.4f}, NDCG@{TOP_N} = {test_ndcg:.4f}, HR@10 = {new_test_hit_10:.4f}, HR@5 = {new_test_hit_5:.4f}")

    with open('output/history/test_base_topk.csv','a') as file:
        file.write(time.strftime("%m-%d %H:%M")+","+f"test,{test_precision:.4f},{test_recall:.4f},{test_f1:.4f},{test_map:.4f},{test_ndcg:.4f},{new_test_hit_10:.4f},{new_test_hit_5:.4f}" + "\n")

    
def test_collab_model(
        args,
        test_loader,
        user_network_stage1,
        item_network_stage1,
        user_review_network,
        item_review_network,
        co_attentions,
        fc_layers_stage2,
    ):
    # ---------- Test ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    user_network_stage1.eval()
    item_network_stage1.eval()
    user_review_network.eval()
    item_review_network.eval()
    co_attentions.eval()
    fc_layers_stage2.eval()

    # These are used to record information in testation.
    test_accs = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    print("-------------------------- TEST --------------------------")
    # Iterate the testation set by batches.
    for batch in tqdm(test_loader):

        # We don't need gradient in testation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():

            # Exacute models       
            user_review_emb, item_review_emb, user_review_mask, item_review_mask, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
            u_batch_size, i_batch_size = len(user_review_emb), len(item_review_emb)
            user_logits = user_network_stage1(user_review_emb.to(args["device"]), user_lda_groups.to(args["device"]))
            item_logits = item_network_stage1(item_review_emb.to(args["device"]), item_lda_groups.to(args["device"]))
            urf = user_review_network(user_logits, user_review_mask.to(args["device"]),  u_batch_size)
            irf = item_review_network(item_logits, item_review_mask.to(args["device"]), i_batch_size)
            w_urf, w_irf = co_attentions(urf, irf)
            user_feature = torch.cat((w_urf, user_mf_emb.to(args["device"])), dim=1)
            item_feature = torch.cat((w_irf, item_mf_emb.to(args["device"])), dim=1)
            fc_input = torch.cat((user_feature, item_feature), dim=1)
            logits = fc_layers_stage2(fc_input)

            # Output after sigmoid is greater than Q will be considered as 1, else 0.
            result_logits = torch.where(logits > 0.5, 1, 0).squeeze(dim=-1)
            labels = labels.to(args["device"])

            # Compute the information for current batch.
            acc = (result_logits == labels.to(args["device"])).float().mean()
            precision = precision_score(labels.cpu(), result_logits.cpu(), zero_division=0)
            recall = recall_score(labels.cpu(), result_logits.cpu())
            f1 = f1_score(labels.cpu(), result_logits.cpu())
            # ndcg = ndcg_score(labels.unsqueeze(dim=-1).cpu(), result_logits.unsqueeze(dim=-1).cpu())

            # Record the information.
            test_accs.append(acc)
            test_precisions.append(precision)
            test_recalls.append(recall)
            test_f1s.append(f1)

    # The average loss and accuracy for entire testation set is the average of the recorded values.
    test_acc = sum(test_accs) / len(test_accs)
    test_precision = sum(test_precisions) / len(test_precisions)
    test_recall = sum(test_recalls) / len(test_recalls)
    test_f1 = sum(test_f1s) / len(test_f1s)

    print(f"[ Test Collab ] acc = {test_acc:.4f}, precision = {test_precision:.4f}, recall = {test_recall:.4f}, f1 = {test_f1:.4f}")
    with open('output/history/test_collab.csv','a') as file:
        file.write(time.strftime("%m-%d %H:%M")+","+f"test,{test_acc:.4f},{test_precision:.4f},{test_recall:.4f},{test_f1:.4f}" + "\n")
                

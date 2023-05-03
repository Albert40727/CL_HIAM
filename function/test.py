import torch
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score


# def mapk(actuals, predicted, k=0):
#   return np.mean([apk(a,p,k) for a,p in product([actuals], [predicted])])

def apk(y_true, y_pred, k_max=0):
    #  Source: https://towardsdatascience.com/mean-average-precision-at-k-map-k-clearly-explained-538d8e032d2
    #  Check if all elements in lists are unique
    if len(set(y_true)) != len(y_true):
        raise ValueError("Values in y_true are not unique")

    if len(set(y_pred)) != len(y_pred):
        raise ValueError("Values in y_pred are not unique")

    if k_max != 0:
        y_pred = y_pred[:k_max]

    correct_predictions = 0
    running_sum = 0

    for i, yp_item in enumerate(y_pred):
        
        k = i+1 # our rank starts at 1
        
        if yp_item in y_true:
            correct_predictions += 1
            running_sum += correct_predictions/k

    return running_sum/len(y_true)

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

    # These are used to record information in test.
    test_maps = []
    test_top_20_hrs = []
    test_top_10_hrs = []
    test_top_5_hrs = []

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


    TOP_N = 20
    top_k_df = predict_incidence_df.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=TOP_N)

    # Save result
    predict_incidence_df.to_csv('output/history/probability_df.csv')
    top_k_df.to_csv('output/history/topk_prediction_df.csv')
    print(predict_incidence_df)
    print(top_k_df)

    # Calculate each score
    for user in predict_incidence_df.index:
        
        # Data for scoring
        user_topk_prediction = top_k_df.loc[user]
        user_label = label_incidence_df.loc[user]
        user_like = user_label[user_label==1].keys()
        user_topk_label = label_incidence_df.loc[user, user_topk_prediction]

        # MAP
        test_map = apk(list(user_like), list(top_k_df.loc[user]), k_max=TOP_N)
        test_maps.append(test_map)

        # Top-20 hit ratio
        user_top_20_hr = len(user_topk_label[user_topk_label==1]) / len(user_like)
        test_top_20_hrs.append(user_top_20_hr)

        # Top-10 hit ratio
        user_top_10_label = user_topk_label[:10]
        user_top_10_hr = len(user_top_10_label[user_top_10_label==1]) / len(user_like)
        test_top_10_hrs.append(user_top_10_hr)

        # Top-5 hit ratio
        user_top_5_label = user_topk_label[:5]
        user_top_5_hr = len(user_top_5_label[user_top_5_label==1]) / len(user_like)
        test_top_5_hrs.append(user_top_5_hr)


    # test_ndcg = sum(test_ndcgs) / len(test_ndcgs)
    test_map = sum(test_maps) / len(test_maps)
    test_top_20_hr = sum(test_top_20_hrs) / len(test_top_20_hrs)
    test_top_10_hr = sum(test_top_10_hrs) / len(test_top_10_hrs)
    test_top_5_hr = sum(test_top_5_hrs) / len(test_top_5_hrs)

    print(f"[ Test base ] MAP@{TOP_N} = {test_map:.4f}, Top-20 = {test_top_20_hr:.4f}, Top-10 = {test_top_10_hr:.4f}, Top-5 = {test_top_5_hr:.4f}")
    with open('output/history/test_base_map_topk.csv','a') as file:
        file.write(time.strftime("%m-%d %H:%M")+","+f"test,{test_map:.4f},{test_top_20_hr:.4f},{test_top_10_hr:.4f},{test_top_5_hr:.4f}" + "\n")

    
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
                

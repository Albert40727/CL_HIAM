import torch
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

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
            user_review_emb, item_review_emb, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
            user_logits = user_network(user_review_emb.to(args["device"]), user_lda_groups.to(args["device"]))
            item_logits = item_network(item_review_emb.to(args["device"]), item_lda_groups.to(args["device"]))
            weighted_user_logits,  weighted_item_logits = co_attention(user_logits, item_logits)
            user_feature = torch.cat((weighted_user_logits, user_mf_emb.to(args["device"])), dim=1)
            item_feature = torch.cat((weighted_item_logits, item_mf_emb.to(args["device"])), dim=1)
            fc_input = torch.cat((user_feature, item_feature), dim=1)
            output_logits = fc_layer(fc_input)

            # We can still compute the loss (but not the gradient).
            labels = labels.to(args["device"])

            # Output after sigmoid is greater than "Q" will be considered as 1, else 0.
            result_logits = torch.where(output_logits > 0.5, 1, 0).squeeze(dim=-1)

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
    test_loss = []
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
            user_review_emb, item_review_emb, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
            u_batch_size, i_batch_size = len(user_review_emb), len(item_review_emb)
            user_logits = user_network_stage1(user_review_emb.to(args["device"]), user_lda_groups.to(args["device"]))
            item_logits = item_network_stage1(item_review_emb.to(args["device"]), item_lda_groups.to(args["device"]))
            urf = user_review_network(user_logits, u_batch_size)
            irf = item_review_network(item_logits, i_batch_size)
            w_urf, w_irf = co_attentions(urf, irf)
            user_feature = torch.cat((w_urf, user_mf_emb.to(args["device"])), dim=1)
            item_feature = torch.cat((w_irf, item_mf_emb.to(args["device"])), dim=1)
            fc_input = torch.cat((user_feature, item_feature), dim=1)
            logits = fc_layers_stage2(fc_input)

            # Output after sigmoid is greater than 0.5 will be considered as 1, else 0.
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
                

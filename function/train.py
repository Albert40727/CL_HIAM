import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score

def train_model(args, train_loader, val_loader, user_network, item_network, co_attention, fc_layer,
                 *, criterion, models_params, optimizer):
    
    # For recording history usage
    t_loss_list, t_acc_list, t_precision_list, t_recall_list, t_f1_list = [], [], [], [], []
    v_loss_list, v_acc_list, v_precision_list, v_recall_list, v_f1_list = [], [], [], [], []
    save_param = {}

    for epoch in range(args["epoch"]):

        n_epochs = args["epoch"]

        user_network.train()
        item_network.train()
        co_attention.train()
        fc_layer.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []
        train_precisions = []
        train_recalls = []
        train_f1s = []

        for batch in tqdm(train_loader):

            # Exacute models
            user_review_emb, item_review_emb, user_review_mask, item_review_mask, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
            user_logits = user_network(user_review_emb.to(args["device"]), user_review_mask.to(args["device"]), user_lda_groups.to(args["device"]))
            item_logits = item_network(item_review_emb.to(args["device"]), item_review_mask.to(args["device"]), item_lda_groups.to(args["device"]))
            weighted_user_logits,  weighted_item_logits = co_attention(user_logits, item_logits)
            user_feature = torch.cat((weighted_user_logits, user_mf_emb.to(args["device"])), dim=1)
            item_feature = torch.cat((weighted_item_logits, item_mf_emb.to(args["device"])), dim=1)
            fc_input = torch.cat((user_feature, item_feature), dim=1)
            logits = fc_layer(fc_input)

            # Sometimes ouput would contain NaN
            if torch.isnan(logits).any() == True:
                print("Warning! Output logits contain NaN")
                logits = torch.nan_to_num(logits, nan=0.0)

            loss = criterion(logits, torch.unsqueeze(labels.to(args["device"]).float(), dim=-1))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(models_params, max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Output after sigmoid is greater than 0.5 will be considered as 1, else 0.
            result_logits = torch.where(logits > 0.5, 1, 0).squeeze(dim=-1)
            labels = labels.to(args["device"])

            # Compute the informations for current batch.
            acc = (result_logits == labels).float().mean()
            precision = precision_score(labels.cpu(), result_logits.cpu(), zero_division=0)
            recall = recall_score(labels.cpu(), result_logits.cpu(), zero_division=0)
            f1 = f1_score(labels.cpu(), result_logits.cpu())
            # ndcg = ndcg_score(labels.unsqueeze(dim=-1).cpu(), result_logits.unsqueeze(dim=-1).cpu())

            # Record the information.
            train_loss.append(loss.item())
            train_accs.append(acc)
            train_precisions.append(precision)
            train_recalls.append(recall)
            train_f1s.append(f1)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        train_precision = sum(train_precisions) / len(train_precisions)
        train_recall = sum(train_recalls) / len(train_recalls)
        train_f1 = sum(train_f1s) / len(train_f1s)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.4f}, precision = {train_precision:.4f}, recall = {train_recall:.4f}, f1 = {train_f1:.4f}")
        with open('output/history/base.csv','a') as file:
            file.write(time.strftime("%m-%d %H:%M")+","+f"train,base,{epoch + 1:03d}/{n_epochs:03d},{train_loss:.5f},{train_acc:.4f},{train_precision:.4f},{train_recall:.4f},{train_f1:.4f}" + "\n")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        user_network.eval()
        item_network.eval()
        co_attention.eval()
        fc_layer.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        valid_precisions = []
        valid_recalls = []
        valid_f1s = []
        # Iterate the validation set by batches.
        for batch in tqdm(val_loader):

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():

                # Exacute models 
                user_review_emb, item_review_emb, user_review_mask, item_review_mask, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
                user_logits = user_network(user_review_emb.to(args["device"]), user_review_mask.to(args["device"]), user_lda_groups.to(args["device"]))
                item_logits = item_network(item_review_emb.to(args["device"]), item_review_mask.to(args["device"]),item_lda_groups.to(args["device"]))
                weighted_user_logits,  weighted_item_logits = co_attention(user_logits, item_logits)
                user_feature = torch.cat((weighted_user_logits, user_mf_emb.to(args["device"])), dim=1)
                item_feature = torch.cat((weighted_item_logits, item_mf_emb.to(args["device"])), dim=1)
                fc_input = torch.cat((user_feature, item_feature), dim=1)
                logits = fc_layer(fc_input)

                # Sometimes ouput would contain NaN
                if torch.isnan(logits).any() == True:
                    print("Warning! Output logits contain NaN")
                    logits = torch.nan_to_num(logits, nan=0.0)

                # We can still compute the loss (but not the gradient).
                labels = labels.to(args["device"])
                loss = criterion(torch.squeeze(logits, dim=-1), labels.float())

                # Output after sigmoid is greater than "Q" will be considered as 1, else 0.
                result_logits = torch.where(logits > 0.5, 1, 0).squeeze(dim=-1)

                # Compute the information for current batch.
                acc = (result_logits == labels).float().mean()
                precision = precision_score(labels.cpu(), result_logits.cpu(), zero_division=0)
                recall = recall_score(labels.cpu(), result_logits.cpu(), zero_division=0)
                f1 = f1_score(labels.cpu(), result_logits.cpu())
                # ndcg = ndcg_score(labels.unsqueeze(dim=-1).cpu(), result_logits.unsqueeze(dim=-1).cpu())

                # Record the information.
                valid_loss.append(loss.item())
                valid_accs.append(acc)
                valid_precisions.append(precision)
                valid_recalls.append(recall)
                valid_f1s.append(f1)
        

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        valid_precision = sum(valid_precisions) / len(valid_precisions)
        valid_recall = sum(valid_recalls) / len(valid_recalls)
        valid_f1 = sum(valid_f1s) / len(valid_f1s)

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.4f}, precision = {valid_precision:.4f}, recall = {valid_recall:.4f}, f1 = {valid_f1:.4f}")
        with open('output/history/base.csv','a') as file:
            file.write(time.strftime("%m-%d %H:%M")+","+f"valid,base,{epoch + 1:03d}/{n_epochs:03d},{valid_loss:.5f},{valid_acc:.4f},{valid_precision:.4f},{valid_recall:.4f},{valid_f1:.4f}" + "\n")

        # Record history
        t_loss_list.append(train_loss)
        t_acc_list.append(train_acc.cpu())
        v_loss_list.append(valid_loss)
        v_acc_list.append(valid_acc.cpu())

        v_f1_list.append(valid_f1)
        if valid_f1 == max(v_f1_list):
            save_param.update({
                'user_review_network' : user_network.state_dict(),
                'item_review_network' : item_network.state_dict(),
                'co_attention' : co_attention.state_dict(),
                'fc_layer' : fc_layer.state_dict(),
                'optimizer': optimizer.state_dict(),
            })

    return t_loss_list, t_acc_list, v_loss_list, v_acc_list, save_param

def draw_loss_curve(train_loss, valid_loss):
    plt.plot(train_loss, color="mediumblue", label="Train", marker='o')
    plt.plot(valid_loss, color="cornflowerblue", label="Valid", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Base Loss Curve")
    plt.savefig('output/plot/base/loss_base_{}.png'.format(time.strftime("%m%d%H%M%S")))
    plt.show()

def draw_acc_curve(train_acc, valid_acc):
    plt.plot(train_acc, color="deeppink", label="Train", marker='o')
    plt.plot(valid_acc, color="pink", label="Valid", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend(loc="upper right")
    plt.title("Base Acc Curve")
    plt.savefig('output/plot/base/acc_base_{}.png'.format(time.strftime("%m%d%H%M%S")))
    plt.show()

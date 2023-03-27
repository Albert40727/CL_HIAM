from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import time

def train_stage2_model(args, 
                       train_loader, 
                       val_loader,                     
                       user_networks, 
                       item_networks,
                       co_attentions, 
                       fc_layers_stage2, 
                       *, 
                       criterion,
                       models_param, 
                       optimizer):
    
    # For plot usage
    t_loss_list_stage2, t_acc_list_stage2 , v_loss_list_stage2, v_acc_list_stage2 = [], [], [], []

    print("-------------------------- STAGE2 START --------------------------")
    for epoch in range(args["epoch_stage2"]):

        n_epochs = args["epoch_stage2"]

        # Frozen stage1 models params
        user_networks[0].eval()
        item_networks[0].eval()

        # Train stage2 models
        user_networks[1].train()
        item_networks[1].train()
        for coa, fc in zip(co_attentions, fc_layers_stage2):
            coa.train()
            fc.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []
        train_precisions = []
        train_recalls = []
        train_f1s = []

        for batch in tqdm(train_loader):

            # Exacute models
            user_review_emb, item_review_emb, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
            u_batch_size, i_batch_size = len(user_review_emb), len(item_review_emb)
            user_logits, _, _, _ = user_networks[0](user_review_emb.to(args["device"]), user_lda_groups.to(args["device"]))
            item_logits, _, _, _ = item_networks[0](item_review_emb.to(args["device"]), item_lda_groups.to(args["device"]))
            urf, urf_1 = user_networks[1](user_logits, u_batch_size)
            urf_2, urf_3 = user_networks[2](user_logits, u_batch_size)
            irf, irf_1 = item_networks[1](item_logits, i_batch_size)
            irf_2, irf_3 = item_networks[2](item_logits, i_batch_size)
            weighted_user_logits,  weighted_item_logits = co_attentions[0](urf, irf)
            weighted_user_logits_1,  weighted_item_logits_1 = co_attentions[1](urf_1, irf_1)
            weighted_user_logits_2,  weighted_item_logits_2 = co_attentions[2](urf_2, irf_2)
            weighted_user_logits_3,  weighted_item_logits_3 = co_attentions[3](urf_3, irf_3)
            user_feature = torch.cat((weighted_user_logits, user_mf_emb.to(args["device"])), dim=1)
            item_feature = torch.cat((weighted_item_logits, item_mf_emb.to(args["device"])), dim=1)
            user_feature_1 = torch.cat((weighted_user_logits_1, user_mf_emb.to(args["device"])), dim=1)
            item_feature_1 = torch.cat((weighted_item_logits_1, item_mf_emb.to(args["device"])), dim=1)
            user_feature_2 = torch.cat((weighted_user_logits_2, user_mf_emb.to(args["device"])), dim=1)
            item_feature_2 = torch.cat((weighted_item_logits_2, item_mf_emb.to(args["device"])), dim=1)
            user_feature_3 = torch.cat((weighted_user_logits_3, user_mf_emb.to(args["device"])), dim=1)
            item_feature_3 = torch.cat((weighted_item_logits_3, item_mf_emb.to(args["device"])), dim=1)
            fc_input = torch.cat((user_feature, item_feature), dim=1)
            fc_input_1 = torch.cat((user_feature_1, item_feature_1), dim=1)
            fc_input_2 = torch.cat((user_feature_2, item_feature_2), dim=1)
            fc_input_3 = torch.cat((user_feature_3, item_feature_3), dim=1)
            logits = fc_layers_stage2[0](fc_input)
            soft_label_1 = fc_layers_stage2[1](fc_input_1)
            soft_label_2 = fc_layers_stage2[2](fc_input_2)
            soft_label_3 = fc_layers_stage2[3](fc_input_3)

            # model.train()  
            loss = ((1-args["trade_off_stage2"])*criterion(logits.reshape(labels.size()), labels.to(args["device"]).float())
                     + args["trade_off_stage2"]*(criterion(logits, soft_label_1) + 
                                                 criterion(logits, soft_label_2) + 
                                                 criterion(logits, soft_label_3)))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(models_param, max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Output after sigmoid is greater than 0.5 will be considered as 1, else 0.
            result_logits = torch.where(logits > 0.5, 1, 0).squeeze(dim=-1)
            labels = labels.to(args["device"])

            # Compute the informations for current batch.
            acc = (result_logits == labels).float().mean()
            precision = precision_score(labels.cpu(), result_logits.cpu(), zero_division=0)
            recall = recall_score(labels.cpu(), result_logits.cpu())
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

        print(f"[ Train stage2 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.4f}, precision = {train_precision:.4f}, recall = {train_recall:.4f}, f1 = {train_f1}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        user_networks[1].eval()
        item_networks[1].eval()
        for coa, fc in zip(co_attentions, fc_layers_stage2):
            coa.eval()
            fc.eval()

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
                user_review_emb, item_review_emb, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
                u_batch_size, i_batch_size = len(user_review_emb), len(item_review_emb)
                user_logits, _, _, _ = user_networks[0](user_review_emb.to(args["device"]), user_lda_groups.to(args["device"]))
                item_logits, _, _, _ = item_networks[0](item_review_emb.to(args["device"]), item_lda_groups.to(args["device"]))
                urf, _ = user_networks[1](user_logits, u_batch_size)
                irf, _ = item_networks[1](item_logits, i_batch_size)
                weighted_user_logits,  weighted_item_logits = co_attentions[0](urf, irf)
                user_feature = torch.cat((weighted_user_logits, user_mf_emb.to(args["device"])), dim=1)
                item_feature = torch.cat((weighted_item_logits, item_mf_emb.to(args["device"])), dim=1)
                fc_input = torch.cat((user_feature, item_feature), dim=1)
                logits = fc_layers_stage2[0](fc_input)


                # We can still compute the loss (but not the gradient).
                loss = criterion(torch.squeeze(logits, dim=1), labels.to(args["device"]).float())

                # Output after sigmoid is greater than 1.3(p>0.8) will be considered as 1, else 0.
                result_logits = torch.where(logits > 0.5, 1, 0).squeeze(dim=-1)
                labels = labels.to(args["device"])

                # Compute the information for current batch.
                result_logits = torch.where(logits > 0.8, 1, 0).squeeze(dim=-1)
                acc = (result_logits == labels.to(args["device"])).float().mean()
                precision = precision_score(labels.cpu(), result_logits.cpu(), zero_division=0)
                recall = recall_score(labels.cpu(), result_logits.cpu())
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

        print(f"[ Valid stage2 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.4f}, precision = {valid_precision:.4f}, recall = {valid_recall:.4f}, f1 = {valid_f1}")

        t_loss_list_stage2.append(train_loss)
        t_acc_list_stage2.append(train_acc.cpu())
        v_loss_list_stage2.append(valid_loss)
        v_acc_list_stage2.append(valid_acc.cpu())

        # if (epoch+1)%5 == 0:
        #     torch.save(user_network_model.state_dict(), f"output/model/user_network_{epoch+1}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt")
        #     torch.save(item_network_model.state_dict(), f"output/model/item_network_{epoch+1}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt")
        #     torch.save(co_attention_network.state_dict(), f"output/model/co_attention_{epoch+1}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt")
        #     torch.save(fc_layer.state_dict(), f"output/model/fc_layer_{epoch+1}.pt")

    print("-------------------------- STAGE2 END --------------------------")

    return t_loss_list_stage2, t_acc_list_stage2, v_loss_list_stage2, v_acc_list_stage2

def draw_loss_curve_stage2(train_loss, valid_loss):
    plt.plot(train_loss, color="blue", label="Train", marker='o')
    plt.plot(valid_loss, color="red", label="Valid", marker='o')
    plt.legend(loc="upper right")
    plt.title("Stage2 Loss Curve")
    plt.savefig('output/plot/collab/loss_stage2_{}.png'.format(time.strftime("%m%d%H%M%S")))
    plt.draw()

def draw_acc_curve_stage2(train_acc, valid_acc):
    plt.plot(train_acc, color="forestgreen", label="Train", marker='o')
    plt.plot(valid_acc, color="gold", label="Valid", marker='o')
    plt.legend(loc="upper right")
    plt.title("Stage2 Acc Curve")
    plt.savefig('output/plot/collab/acc_stage2_{}.png'.format(time.strftime("%m%d%H%M%S")))
    plt.draw()
    return
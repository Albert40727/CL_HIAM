from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import datetime

def train_model(args, train_loader, val_loader, user_network_model, item_network_model, co_attention_network, fc_layer,
                 *, criterion, models_params, optimizer):
    
    t_loss_list, t_acc_list, t_precision_list, t_recall_list, t_f1_list = [], [], [], [], []
    v_loss_list, v_acc_list, v_precision_list, v_recall_list, v_f1_list = [], [], [], [], []

    for epoch in range(args["epoch"]):

        n_epochs = args["epoch"]

        user_network_model.train()
        item_network_model.train()
        co_attention_network.train()
        fc_layer.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []
        train_precisions = []
        train_recalls = []
        train_f1s = []

        for batch in tqdm(train_loader):

            # Exacute models
            user_review_emb, item_review_emb, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
            user_logits = user_network_model(user_review_emb.to(args["device"]), user_lda_groups.to(args["device"]))
            item_logits = item_network_model(item_review_emb.to(args["device"]), item_lda_groups.to(args["device"]))
            weighted_user_logits,  weighted_item_logits = co_attention_network(user_logits, item_logits)
            user_feature = torch.cat((weighted_user_logits, user_mf_emb.to(args["device"])), dim=1)
            item_feature = torch.cat((weighted_item_logits, item_mf_emb.to(args["device"])), dim=1)
            fc_input = torch.cat((user_feature, item_feature), dim=1)
            output_logits = fc_layer(fc_input)

            # model.train()  
            loss = criterion(torch.squeeze(output_logits, dim=1), labels.to(args["device"]).float())

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(models_params, max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Output after sigmoid is greater than 0.5 will be considered as 1, else 0.
            result_logits = torch.where(output_logits > 0.5, 1, 0).squeeze(dim=-1)
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

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.4f}, precision = {train_precision:.4f}, recall = {train_recall:.4f}, f1 = {train_f1}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        user_network_model.eval()
        item_network_model.eval()
        co_attention_network.eval()
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
                user_review_emb, item_review_emb, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, labels = batch
                user_logits = user_network_model(user_review_emb.to(args["device"]), user_lda_groups.to(args["device"]))
                item_logits = item_network_model(item_review_emb.to(args["device"]), item_lda_groups.to(args["device"]))
                weighted_user_logits,  weighted_item_logits = co_attention_network(user_logits.to(args["device"]), item_logits.to(args["device"]))
                user_feature = torch.cat((weighted_user_logits.to(args["device"]), user_mf_emb.to(args["device"])), dim=1)
                item_feature = torch.cat((weighted_item_logits.to(args["device"]), item_mf_emb.to(args["device"])), dim=1)
                fc_input = torch.cat((user_feature, item_feature), dim=1)
                output_logits = fc_layer(fc_input)

                # We can still compute the loss (but not the gradient).
                loss = criterion(torch.squeeze(output_logits, dim=1), labels.to(args["device"]).float())

                # Output after sigmoid is greater than 1.3(p>0.8) will be considered as 1, else 0.
                result_logits = torch.where(output_logits > 0.8, 1, 0).squeeze(dim=-1)
                labels = labels.to(args["device"])

                # Compute the information for current batch.
                result_logits = torch.where(output_logits > 0.8, 1, 0).squeeze(dim=-1)
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

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.4f}, precision = {valid_precision:.4f}, recall = {valid_recall:.4f}, f1 = {valid_f1}")

        t_loss_list.append(train_loss)
        t_acc_list.append(train_acc.cpu())
        v_loss_list.append(valid_loss)
        v_acc_list.append(valid_acc.cpu())

        if (epoch+1)%5 == 0:
            torch.save(user_network_model.state_dict(), f"output/model/user_network_{epoch+1}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt")
            torch.save(item_network_model.state_dict(), f"output/model/item_network_{epoch+1}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt")
            torch.save(co_attention_network.state_dict(), f"output/model/co_attention_{epoch+1}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt")
            torch.save(fc_layer.state_dict(), f"output/model/fc_layer_{epoch+1}.pt")

    return t_loss_list, t_acc_list, v_loss_list, v_acc_list

def draw_loss_curve(train_loss, valid_loss):
    plt.plot(train_loss, color="blue", label="Train")
    plt.plot(valid_loss, color="red", label="Valid")
    plt.legend(loc="upper right")
    plt.title("Loss Curve")
    plt.savefig(f'output/plot/loss_plot_{datetime.datetime.now()}.png')
    plt.show()

def draw_acc_curve(train_loss, valid_loss):
    plt.plot(train_loss, color="blue", label="Train")
    plt.plot(valid_loss, color="red", label="Valid")
    plt.legend(loc="upper right")
    plt.title("Acc Curve")
    plt.savefig(f'output/plot/acc_plot_{datetime.datetime.now()}.png')
    plt.show()
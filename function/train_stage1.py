import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def train_stage1_model(args, 
                       train_loader, 
                       val_loader,                     
                       user_network, 
                       item_network,
                       user_fc_layer_stage1,
                       item_fc_layer_stage1,
                       *, 
                       criterions,
                       models_params, 
                       optimizers):
    
    # For recording history usage
    t_user_loss_list_stage1, t_user_acc_list_stage1, t_item_loss_list_stage1, t_item_acc_list_stage1 = [], [], [], []
    v_user_loss_list_stage1, v_user_acc_list_stage1, v_item_loss_list_stage1, v_item_acc_list_stage1, v_user_f1_list_stage1, v_item_f1_list_stage1 = [], [], [], [], [], []
    save_param = {}

    print("-------------------------- STAGE1 START --------------------------")
    # Stage 1 training
    for epoch in range(args["epoch_stage1"]):

        n_epochs = args["epoch_stage1"]
        
        user_network.train()
        item_network.train()
        user_fc_layer_stage1.train()
        item_fc_layer_stage1.train()
        
        # These are used to record information in training.
        user_train_loss_stage1, user_train_accs_stage1, user_train_precisions_stage1, user_train_recalls_stage1, user_train_f1s_stage1 = [], [], [], [], []
        item_train_loss_stage1, item_train_accs_stage1, item_train_precisions_stage1, item_train_recalls_stage1, item_train_f1s_stage1 = [], [], [], [], []

        for batch in tqdm(train_loader[0]):
            # Exacute user stage1 models
            user_review_emb, user_lda_groups, user_mf_emb, user_labels = batch
            loss, acc, precision, recall, f1 = \
            batch_train_stage1(args, user_review_emb, user_lda_groups, user_labels,
                               target = "user",
                               network = user_network, 
                               fc_layers = user_fc_layer_stage1,
                               criterion = criterions[0], 
                               models_params = models_params[0], 
                               optimizers = optimizers[0])
        
            # Record the user-network information.
            user_train_loss_stage1.append(loss)
            user_train_accs_stage1.append(acc)
            user_train_precisions_stage1.append(precision)
            user_train_recalls_stage1.append(recall)
            user_train_f1s_stage1.append(f1)
        
        for batch in tqdm(train_loader[1]):
            # Exacute item stage1 models
            item_review_emb, item_lda_groups, item_mf_emb, item_labels = batch
            loss, acc, precision, recall, f1 = \
            batch_train_stage1(args, item_review_emb, item_lda_groups, item_labels,
                               target = "item",
                               network = item_network,
                               fc_layers = item_fc_layer_stage1, 
                               criterion = criterions[1], 
                               models_params = models_params[1], 
                               optimizers = optimizers[1])
            
            # Record the item-network information.
            item_train_loss_stage1.append(loss)
            item_train_accs_stage1.append(acc)
            item_train_precisions_stage1.append(precision)
            item_train_recalls_stage1.append(recall)
            item_train_f1s_stage1.append(f1)
            
        # The average loss and accuracy of the training set is the average of the recorded values.
        user_train_loss, user_train_acc, user_train_precision, user_train_recall, user_train_f1 = \
        epoch_info(user_train_loss_stage1, 
                   user_train_accs_stage1, 
                   user_train_precisions_stage1,
                   user_train_recalls_stage1,
                   user_train_f1s_stage1,
                   mode = "Train",
                   target = "user",
                   epoch = epoch,
                   n_epochs = n_epochs)
        
        item_train_loss, item_train_acc, item_train_precision, item_train_recall, item_train_f1 = \
        epoch_info(item_train_loss_stage1, 
                   item_train_accs_stage1, 
                   item_train_precisions_stage1,
                   item_train_recalls_stage1,
                   item_train_f1s_stage1,
                   mode = "Train",
                   target = "item",
                   epoch = epoch,
                   n_epochs = n_epochs)

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        user_network.eval()
        item_network.eval()
        user_fc_layer_stage1.eval()
        item_fc_layer_stage1.eval()

        # These are used to record information in validation.
        user_val_loss_stage1, user_val_accs_stage1, user_val_precisions_stage1, user_val_recalls_stage1, user_val_f1s_stage1 = [], [], [], [], []
        item_val_loss_stage1, item_val_accs_stage1, item_val_precisions_stage1, item_val_recalls_stage1, item_val_f1s_stage1 = [], [], [], [], []
        
        # Iterate the validation set by batches.
        for batch in tqdm(val_loader[0]):
            # User stage1 model
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                user_review_emb, user_lda_groups, user_mf_emb, user_labels = batch
                
                loss, acc, precision, recall, f1 = \
                batch_val_stage1(args, user_review_emb, user_lda_groups, user_labels,
                                 target = "user",
                                 network = user_network,
                                 fc_layers = user_fc_layer_stage1, 
                                 criterion = criterions[0])
                # Record the usernetwork information.
                user_val_loss_stage1.append(loss)
                user_val_accs_stage1.append(acc)
                user_val_precisions_stage1.append(precision)
                user_val_recalls_stage1.append(recall)
                user_val_f1s_stage1.append(f1)

        
        for batch in tqdm(val_loader[1]):
            # Item stage1 model
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():     
                item_review_emb, item_lda_groups, user_mf_emb, item_labels = batch 

                loss, acc, precision, recall, f1 = \
                batch_val_stage1(args, item_review_emb, item_lda_groups, item_labels,
                                 target = "item",
                                 network = item_network,
                                 fc_layers = item_fc_layer_stage1, 
                                 criterion = criterions[1])
                
                # Record the itemnetwork information.
                item_val_loss_stage1.append(loss)
                item_val_accs_stage1.append(acc)
                item_val_precisions_stage1.append(precision)
                item_val_recalls_stage1.append(recall)
                item_val_f1s_stage1.append(f1)
                
        # The average loss and accuracy for entire validation set is the average of the recorded values.
        user_val_loss, user_val_acc, user_val_precision, user_val_recall, user_val_f1 = \
        epoch_info(user_val_loss_stage1, 
                   user_val_accs_stage1, 
                   user_val_precisions_stage1,
                   user_val_recalls_stage1,
                   user_val_f1s_stage1,
                   mode = "Valid",
                   target = "user",
                   epoch = epoch,
                   n_epochs = n_epochs)
        
        item_val_loss, item_val_acc, item_val_precision, item_val_recall, item_val_f1 = \
        epoch_info(item_val_loss_stage1, 
                   item_val_accs_stage1, 
                   item_val_precisions_stage1,
                   item_val_recalls_stage1,
                   item_val_f1s_stage1,
                   mode = "Valid",
                   target = "item",
                   epoch = epoch,
                   n_epochs = n_epochs)

        # Record history
        t_user_loss_list_stage1.append(user_train_loss)
        t_user_acc_list_stage1.append(user_train_acc.cpu())
        t_item_loss_list_stage1.append(item_train_loss)
        t_item_acc_list_stage1.append(item_train_acc.cpu())

        v_user_loss_list_stage1.append(user_val_loss)
        v_user_acc_list_stage1.append(user_val_acc.cpu())
        v_item_loss_list_stage1.append(item_val_loss)
        v_item_acc_list_stage1.append(item_val_acc.cpu())

        v_user_f1_list_stage1.append(user_val_f1)
        v_item_f1_list_stage1.append(item_val_f1)

        # Param need to be saved according to highest f1 of val
        if user_val_f1 == max(v_user_f1_list_stage1):
            print("Update User network save_param !")
            save_param.update({
                'user_network_stage1': user_network.state_dict(),
                'user_fc_layer_stage1' : user_fc_layer_stage1.state_dict(),
                'user_optimizer_stage1' : optimizers[0].state_dict(),
                })

        if item_val_f1 == max(v_item_f1_list_stage1):
            print("Update Item network save_param !")
            save_param.update({
                'item_network_stage1': item_network.state_dict(),
                'item_fc_layer_stage1' : item_fc_layer_stage1.state_dict(),
                'item_optimizer_stage1' :  optimizers[1].state_dict(),
                })

    print("-------------------------- STAGE1 END --------------------------")

    return t_user_loss_list_stage1, t_user_acc_list_stage1, t_item_loss_list_stage1, t_item_acc_list_stage1,\
           v_user_loss_list_stage1, v_user_acc_list_stage1, v_item_loss_list_stage1, v_item_acc_list_stage1, save_param

def batch_train_stage1(args, review_emb, lda_groups, labels, *, 
                       target, network, fc_layers, criterion, models_params, optimizers):

    arv, arv_1, arv_2, arv_3 = network(review_emb.to(args["device"]), lda_groups.to(args["device"]))
    logits, soft_label_1, soft_label_2, soft_label_3 = fc_layers(arv, arv_1, arv_2, arv_3)

    if torch.isnan(torch.stack((logits, soft_label_1, soft_label_2, soft_label_3))).any() == True:
        print(f"Warning! {target} network's output logits contain NaN")
        logits = torch.nan_to_num(logits, nan=0.0)
    
    loss =  (args["trade_off_stage1"]*criterion(logits.reshape(labels.size()), labels.to(args["device"]).float())
                + (1-args["trade_off_stage1"])*(criterion(logits, (soft_label_1 + soft_label_2 + soft_label_3)/3))) +\
            (args["trade_off_stage1"]*criterion(soft_label_1.reshape(labels.size()), labels.to(args["device"]).float())
                + (1-args["trade_off_stage1"])*(criterion(soft_label_1, (logits + soft_label_2 + soft_label_3)/3))) +\
            (args["trade_off_stage1"]*criterion(soft_label_2.reshape(labels.size()), labels.to(args["device"]).float())
                + (1-args["trade_off_stage1"])*(criterion(soft_label_2, (logits + soft_label_1 + soft_label_3)/3))) +\
            (args["trade_off_stage1"]*criterion(soft_label_3.reshape(labels.size()), labels.to(args["device"]).float())
                + (1-args["trade_off_stage1"])*(criterion(soft_label_3, (logits + soft_label_1 + soft_label_2)/3)))
    
    loss =  criterion(logits.reshape(labels.size()), labels.to(args["device"]).float())
            
    # Gradients stored in the parameters in the previous step should be cleared out first.
    optimizers.zero_grad()

    # Compute the gradients for parameters.
    loss.backward()

    # Clip the gradient norms for stable training.
    nn.utils.clip_grad_norm_(models_params, max_norm=1)

    # Update the parameters with computed gradients.
    optimizers.step()

    # Output after sigmoid is greater than "Q" will be considered as 1, else 0.
    result_logits = torch.where(logits > 0.5, 1, 0).reshape(labels.shape)
    labels = labels.to(args["device"])

    # Compute the informations for current batch.
    acc = (result_logits == labels).float().mean()
    precision = precision_score(labels.cpu(), result_logits.cpu(), zero_division=0, average="samples")
    recall = recall_score(labels.cpu(), result_logits.cpu(), zero_division=0, average="samples")
    f1 = f1_score(labels.cpu(), result_logits.cpu(), zero_division=0, average="samples")

    return loss.item(), acc, precision, recall, f1

def batch_val_stage1(args, review_emb, lda_groups, labels, 
                     *, target, network, fc_layers, criterion):
    # Exacute models 
    arv = network(review_emb.to(args["device"]), lda_groups.to(args["device"]))
    logits = fc_layers(arv)

    if torch.isnan(logits).any() == True:
        print("Warning! Output logits contain NaN")
        logits = torch.nan_to_num(logits, nan=0.0)

    # We can still compute the loss (but not the gradient).
    loss = criterion(logits.reshape(labels.size()), labels.to(args["device"]).float())

    # Output after sigmoid is greater than 0.5 will be considered as 1, else 0.
    result_logits = torch.where(logits > 0.5, 1, 0).reshape(labels.shape)
    labels = labels.to(args["device"])

    # Compute the information for current batch.
    acc = (result_logits == labels).float().mean()
    precision = precision_score(labels.cpu(), result_logits.cpu(), zero_division=0, average="samples")
    recall = recall_score(labels.cpu(), result_logits.cpu(), zero_division=0, average="samples")
    f1 = f1_score(labels.cpu(), result_logits.cpu(), zero_division=0, average="samples")

    return loss.item(), acc, precision, recall, f1   

def epoch_info(loss, accs, precisions, recalls, f1s, *, mode, target, epoch, n_epochs):
    # Calculate all the info and print
    mean_loss = sum(loss) / len(loss)
    mean_acc = sum(accs) / len(accs)
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    mean_f1 = sum(f1s) / len(f1s)
    print(f"[ {mode} {target}-stage1 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {mean_loss:.5f}, acc = {mean_acc:.4f}, precision = {mean_precision:.4f}, recall = {mean_recall:.4f}, f1 = {mean_f1:.4f}")

    with open(f'output/history/{target}_stage1.csv','a') as file:
        file.write(time.strftime("%m-%d %H:%M")+","+f"{mode},{target}-stage1,{epoch + 1:03d}/{n_epochs:03d},{mean_loss:.5f},{mean_acc:.4f},{mean_precision:.4f},{mean_recall:.4f},{mean_f1:.4f}" + "\n")
    
    return mean_loss, mean_acc, mean_precision, mean_recall, mean_f1

def draw_loss_curve_stage1(u_train_loss, u_valid_loss, i_train_loss, i_valid_loss):
    plt.plot(u_train_loss, color="mediumblue", label="user-train", marker='o')
    plt.plot(u_valid_loss, color="cornflowerblue", label="user-valid", marker='o')
    plt.plot(i_train_loss, color="deeppink", label="item-train", marker='o')
    plt.plot(i_valid_loss, color="pink", label="item-valid", marker='o')
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Stage1 Loss Curve")
    plt.savefig('output/plot/collab/loss_stage1_{}.png'.format(time.strftime("%m%d%H%M%S")))
    plt.show()

def draw_acc_curve_stage1(u_train_acc, u_valid_acc, i_train_acc, i_valid_acc):
    plt.plot(u_train_acc, color="mediumblue", label="user-train", marker='o')
    plt.plot(u_valid_acc, color="cornflowerblue", label="user-valid", marker='o')
    plt.plot(i_train_acc, color="deeppink", label="item-train", marker='o')
    plt.plot(i_valid_acc, color="pink", label="item-valid", marker='o')
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title(f"Stage1 Acc Curve")
    plt.savefig('output/plot/collab/acc_stage1_{}.png'.format(time.strftime("%m%d%H%M%S")))
    plt.show()

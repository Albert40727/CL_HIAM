import torch
import torch.nn as nn
import time
from model.hian import HianModel
from model.fc_layer import FcLayer
from model.fc_layer_stage1 import FcLayerStage1
from model.fc_layer_stage2 import FcLayerStage2
from model.co_attention import CoattentionNet
from model.co_attention_stage2 import CoattentionNetStage2
from model.hian_cl_stage1 import HianCollabStage1
from model.review_net_stage2 import ReviewNetworkStage2
from model.bp_gate import BackPropagationGate
from torch.utils.data import DataLoader
from function.review_dataset import ReviewDataset, ReviewDataseStage1
from function.train import train_model, draw_acc_curve, draw_loss_curve
from function.train_stage1 import train_stage1_model, draw_acc_curve_stage1, draw_loss_curve_stage1
from function.train_stage2 import train_stage2_model, draw_acc_curve_stage2, draw_loss_curve_stage2
from function.test import test_model, test_collab_model

                                                                   
def main(**args):

    # Dataset/loader
    train_dataset = ReviewDataset(args, mode="train")
    val_dataset = ReviewDataset(args, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=True)
    
    # Traing base model
    if not args["collab_learning"] and args["train"]:
        # Init model
        user_network_model = HianModel(args).to(device)
        item_network_model = HianModel(args).to(device)
        co_attention = CoattentionNet(args, args["co_attention_emb_dim"]).to(device)
        fc_layer = FcLayer().to(device)

        # Loss criteria
        # class_weight = torch.tensor(args["class_weight"]).to(args["device"])
        # criterion = nn.CrossEntropyLoss(weight=class_weight)
        criterion = nn.BCELoss()
        params = list(user_network_model.parameters()) + list(item_network_model.parameters()) + list(co_attention.parameters()) + list(fc_layer.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)

        # Training 
        train_loss, train_acc, val_loss, val_acc, save_param = \
        train_model(
            args, 
            train_loader, 
            val_loader, 
            user_network_model, 
            item_network_model, 
            co_attention, 
            fc_layer, 
            criterion = criterion, 
            models_params = params, 
            optimizer = optimizer)

        # Save model
        BASE_PATH = args["model_save_path_base"] + "model_base_{}.pt".format(time.strftime("%m%d%H%M%S"))
        torch.save(save_param, BASE_PATH)
        
        # Plot loss & acc curves
        draw_loss_curve(train_loss, val_loss)
        draw_acc_curve(train_acc, val_acc)

    # Train collab model
    elif args["collab_learning"] and args["train"]:
        
        # Create stage1 dataset and loader
        train_dataset_stage1 = ReviewDataseStage1(args, mode="train")
        val_dataset_stage1 = ReviewDataseStage1(args, mode="val")
        train_loader_stage1 = DataLoader(train_dataset_stage1, batch_size=args["batch_size"], shuffle=True)
        val_loader_stage1 = DataLoader(val_dataset_stage1, batch_size=args["batch_size"], shuffle=True)

        # Init model
        # Stage 1
        user_network_stage1 = HianCollabStage1(args).to(device)
        item_network_stage1 = HianCollabStage1(args).to(device)
        user_fc_layer_stage1 = FcLayerStage1().to(device)
        item_fc_layer_stage1 = FcLayerStage1().to(device)

        # Stage2
        user_review_network = ReviewNetworkStage2(args).to(device)
        item_review_network = ReviewNetworkStage2(args).to(device)
        co_attentions = CoattentionNetStage2(args, args["co_attention_emb_dim"]).to(device)
        fc_layers_stage2 = FcLayerStage2().to(device)

        # Back propagation gate
        bp_gate = BackPropagationGate()

        # Loss criteria
        user_criterion_stage1 = nn.BCELoss()
        item_criterion_stage1 = nn.BCELoss()
        criterion_stage2 = nn.BCELoss()

        # Parameters
        user_params_stage1 = (list(user_network_stage1.parameters()) + 
                              list(user_fc_layer_stage1.parameters()))
                              
        item_params_stage1 = (list(item_network_stage1.parameters()) +
                              list(item_fc_layer_stage1.parameters()))
        
        params_stage2 = (list(user_review_network.parameters()) + 
                         list(item_review_network.parameters()) +
                         list(co_attentions.parameters()) + 
                         list(fc_layers_stage2.parameters()))
        
        user_optimizer_stage1 = torch.optim.Adam(user_params_stage1, lr=1e-4, weight_decay=1e-5)
        item_optimizer_stage1 =  torch.optim.Adam(item_params_stage1, lr=1e-4, weight_decay=1e-5)
        optimizer_stage2 = torch.optim.Adam(params_stage2, lr=1e-4, weight_decay=1e-5)

        # Training 
        # Stage1 
        (t_user_loss_stage1, t_user_acc_stage1, t_item_loss_stage1, t_item_acc_stage1,
          v_user_loss_stage1, v_user_acc_stage1, v_item_loss_stage1, v_item_acc_stage1, save_param_stage1) = \
        train_stage1_model(
            args,                                                           
            train_loader_stage1,
            val_loader_stage1,
            user_network_stage1,
            item_network_stage1, 
            user_fc_layer_stage1,
            item_fc_layer_stage1,
            criterions = [user_criterion_stage1, item_criterion_stage1], 
            models_params = [user_params_stage1, item_params_stage1], 
            optimizers = [user_optimizer_stage1, item_optimizer_stage1])
        
        # Save stage1 model
        # Make sure you've change the STAGE1_PATH if you only want to train stage2
        STAGE1_PATH = args["model_save_path_cl"] + "model_cl_stage1_{}.pt".format(time.strftime("%m%d%H%M%S"))
        torch.save(save_param_stage1, STAGE1_PATH)

        # Load stage1 model before training stage2
        user_network_stage1.load_state_dict(save_param_stage1["user_network_stage1"])
        item_network_stage1.load_state_dict(save_param_stage1["item_network_stage1"])

        # Stage2
        t_loss_stage2, t_acc_stage2, v_loss_stage2, v_acc_stage2, save_param_stage2 = \
        train_stage2_model(
            args,                                                           
            train_loader,
            val_loader,
            user_network_stage1,
            item_network_stage1, 
            user_review_network,
            item_review_network, 
            co_attentions, 
            fc_layers_stage2, 
            bp_gate = bp_gate,
            criterion = criterion_stage2, 
            models_param = params_stage2, 
            optimizer = optimizer_stage2)
        
        # Save stage2 model
        STAGE2_PATH = args["model_save_path_cl"] + "model_cl_stage2_{}.pt".format(time.strftime("%m%d%H%M%S"))
        torch.save(save_param_stage2, STAGE2_PATH)
        
        # Plot stage1 loss & acc
        draw_loss_curve_stage1(t_user_loss_stage1, v_user_loss_stage1, t_item_loss_stage1, v_item_loss_stage1)
        draw_acc_curve_stage1(t_user_acc_stage1, v_user_acc_stage1, t_item_acc_stage1, v_item_acc_stage1)
        
        # Plot stage2 loss & acc
        draw_loss_curve_stage2(t_loss_stage2, v_loss_stage2)
        draw_acc_curve_stage2(t_acc_stage2, v_acc_stage2)

    # Test model
    if not args["collab_learning"] and args["test"]:
        if args["train"]:
            checkpoint = torch.load(BASE_PATH)
        else:
            SPEC_PATH = args["model_save_path_base"] + "model_base_0422060354.pt" # Specify .pt you want to load
            checkpoint = torch.load(SPEC_PATH)

        # Init dataset and loader    
        test_dataset = ReviewDataset(args, mode="test")
        test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=True)

        # Init model
        user_network_model = HianModel(args).to(device)
        item_network_model = HianModel(args).to(device)
        co_attention = CoattentionNet(args, args["co_attention_emb_dim"]).to(device)
        fc_layer = FcLayer().to(device)
        user_network_model.load_state_dict(checkpoint["user_review_network"])
        item_network_model.load_state_dict(checkpoint["item_review_network"])
        co_attention.load_state_dict(checkpoint["co_attention"])
        fc_layer.load_state_dict(checkpoint["fc_layer"])

        # Exacute test
        test_model(
            args,
            test_loader,
            user_network_model,
            item_network_model,
            co_attention,
            fc_layer,
        )

    elif args["collab_learning"] and args["test"]:
        if args["train"]:
            checkpoint_stage1 = torch.load(STAGE1_PATH)
            checkpoint_stage2 = torch.load(STAGE2_PATH)
        else:
            SPEC_PATH_STAGE1 = args["model_save_path_cl"] + "model_cl_stage1_0412061332.pt" # Specify .pt you want to load
            SPEC_PATH_STAGE2 = args["model_save_path_cl"] + "model_cl_stage2_0412121305.pt" # Specify .pt you want to load
            checkpoint_stage1 = torch.load(SPEC_PATH_STAGE1)
            checkpoint_stage2 = torch.load(SPEC_PATH_STAGE2)

        # Init dataset and loader    
        test_dataset = ReviewDataset(args, mode="test")
        test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=True)

        # Init model
        user_network_stage1 = HianCollabStage1(args).to(device)
        item_network_stage1 = HianCollabStage1(args).to(device)
        user_review_network = ReviewNetworkStage2(args).to(device)
        item_review_network = ReviewNetworkStage2(args).to(device)
        co_attentions = CoattentionNetStage2(args, args["co_attention_emb_dim"]).to(device)
        fc_layers_stage2 = FcLayerStage2().to(device)
        user_network_stage1.load_state_dict(checkpoint_stage1["user_network_stage1"])
        item_network_stage1.load_state_dict(checkpoint_stage1["item_network_stage1"])
        user_review_network.load_state_dict(checkpoint_stage2["user_review_network"])
        item_review_network.load_state_dict(checkpoint_stage2["item_review_network"])
        co_attentions.load_state_dict(checkpoint_stage2["co_attention_stage2"])
        fc_layers_stage2.load_state_dict(checkpoint_stage2["fc_layer_stage2"])

        # Exacute test
        test_collab_model(
            args,
            test_loader,
            user_network_stage1,
            item_network_stage1,
            user_review_network,
            item_review_network,
            co_attentions,
            fc_layers_stage2,
        )


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = {
        "device" : device,
        "train": True, # Turn off to test only 
        "test": True, # Turn off to train only 
        "train_data_dir" : r'../data/train_df.pkl',
        "val_data_dir" : r'../data/val_df.pkl',
        "test_data_dir" : r'../data/test_df.pkl',
        "user_data_dir" : r'../data/user_emb/',
        "item_data_dir" : r'../data/item_emb/',
        "user_mf_data_dir" : r'../data/train_user_mf_emb.pkl',
        "item_mf_data_dir" : r'../data/train_item_mf_emb.pkl',
        "model_save_path_base" : r"output/model/base/",
        "model_save_path_cl" : r"output/model/collab/",
        "max_word" : 25,
        "max_sentence" : 10,
        "max_review_user" : 10,
        "max_review_item" : 50,
        "emb_dim" : 768,
        "co_attention_emb_dim" : 512,
        "mf_emb_dim" : 128,
        "lda_group_num": 6, # Include default 0 group. 
        "word_cnn_ksize" : 5,   # odd number 
        "sentence_cnn_ksize" : 3,   # odd number 
        "batch_size": 32,
        "collab_learning": True,
        "epoch" : 50, # when "collab_learning" is False
        "epoch_stage1" : 25, # when "collab_learning" is True
        "epoch_stage2" : 25, # when "collab_learning" is True
        "trade_off_stage1": 0.3, # when "collab_learning" is True
        "trade_off_stage2": 0.3, # when "collab_learning" is True
        # "class_weight" : [1.25, 5]  weights for classes when computing loss 
    }

    print("Device: ", device)
    print("Collab: ", args["collab_learning"])
    if args["collab_learning"]:
        print("Stage1 Epochs: ", args["epoch_stage1"], "Trade-off: ", args["trade_off_stage1"])
        print("Stage2 Epochs: ", args["epoch_stage2"], "Trade-off: ", args["trade_off_stage2"])
    else:
        print("Epoch: ", args["epoch"])

    main(**args)
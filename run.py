import time
import torch
import torch.nn as nn
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
from function.review_dataset import ReviewDataset, UserReviewDataseStage1, ItemReviewDataseStage1
from function.train import train_model, draw_acc_curve, draw_loss_curve
from function.train_stage1 import train_stage1_model, draw_acc_curve_stage1, draw_loss_curve_stage1
from function.train_stage2 import train_stage2_model, draw_acc_curve_stage2, draw_loss_curve_stage2
from function.test import test_model, test_model_topk, test_collab_model, test_collab_model_topk

                                                                   
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
        criterion = nn.BCELoss()
        params = list(user_network_model.parameters()) + list(item_network_model.parameters()) + list(co_attention.parameters()) + list(fc_layer.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

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

    # Train collab model
    elif args["collab_learning"] and args["train"]:
        
        # Create stage1 dataset and loader
        user_train_dataset_stage1 = UserReviewDataseStage1(args,mode="train")
        user_val_dataset_stage1 = UserReviewDataseStage1(args, mode="val")
        item_train_dataset_stage1 = ItemReviewDataseStage1(args,mode="train")
        item_val_dataset_stage1 = ItemReviewDataseStage1(args, mode="val")
        
        user_train_loader_stage1 = DataLoader(user_train_dataset_stage1, batch_size=args["batch_size"], shuffle=True)
        user_val_loader_stage1 = DataLoader(user_val_dataset_stage1, batch_size=args["batch_size"], shuffle=True)
        item_train_loader_stage1 = DataLoader(item_train_dataset_stage1, batch_size=args["batch_size"], shuffle=True)
        item_val_loader_stage1 = DataLoader(item_val_dataset_stage1, batch_size=args["batch_size"], shuffle=True)

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
        user_criterion_stage1 = nn.CrossEntropyLoss()
        item_criterion_stage1 = nn.CrossEntropyLoss()
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
        
        user_optimizer_stage1 = torch.optim.Adam(user_params_stage1, lr=1e-5, weight_decay=1e-6) # lr can't be to big. Causing NaN output!!!
        item_optimizer_stage1 =  torch.optim.Adam(item_params_stage1, lr=1e-5, weight_decay=1e-6) # lr can't be to big. Causing NaN output!!!
        optimizer_stage2 = torch.optim.Adam(params_stage2, lr=1e-3, weight_decay=1e-4)

        # Training 
        # Stage1 
        (t_user_loss_stage1, t_user_acc_stage1, t_item_loss_stage1, t_item_acc_stage1,
          v_user_loss_stage1, v_user_acc_stage1, v_item_loss_stage1, v_item_acc_stage1, save_param_stage1) = \
        train_stage1_model(
            args,                                                           
            [user_train_loader_stage1, item_train_loader_stage1],
            [user_val_loader_stage1, item_val_loader_stage1],
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
        

    # Test model
    if not args["collab_learning"] and args["test"]:
        if args["train"]:
            checkpoint = torch.load(BASE_PATH)
            print("Apply trained model param of the highest F1 score.")
        else:
            SPEC_PATH = args["model_save_path_base"] + "model_base_0525190917.pt" # Specify .pt you want to load
            checkpoint = torch.load(SPEC_PATH)
            print(f"Apply specified model param.")

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
        test_model_topk(
            args,
            test_loader,
            user_network_model,
            item_network_model,
            co_attention,
            fc_layer,
        )

    # testing collab model
    elif args["collab_learning"] and args["test"]:
        if args["train"]:
            checkpoint_stage1 = torch.load(STAGE1_PATH)
            checkpoint_stage2 = torch.load(STAGE2_PATH)
            print("Apply trained model param of the highest F1 score.")
        else:
            SPEC_PATH_STAGE1 = args["model_save_path_cl"] + "model_cl_stage1_0529041619.pt" # Specify .pt you want to load
            SPEC_PATH_STAGE2 = args["model_save_path_cl"] + "model_cl_stage2_0529133812.pt" # Specify .pt you want to load
            checkpoint_stage1 = torch.load(SPEC_PATH_STAGE1)
            checkpoint_stage2 = torch.load(SPEC_PATH_STAGE2)
            print(f"Apply specified collab model param.")

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
        test_collab_model_topk(
            args,
            test_loader,
            user_network_stage1,
            item_network_stage1,
            user_review_network,
            item_review_network,
            co_attentions,
            fc_layers_stage2,
        )

    # Be warning that plot will block the process. Therefore, should be put at the final process.
    if not args["collab_learning"] and args["train"]:
        # Plot loss & acc curves
        draw_loss_curve(train_loss, val_loss)
        draw_acc_curve(train_acc, val_acc)
    elif args["collab_learning"] and args["train"]:
        # Plot stage1 loss & acc
        draw_loss_curve_stage1(t_user_loss_stage1, v_user_loss_stage1, t_item_loss_stage1, v_item_loss_stage1)
        draw_acc_curve_stage1(t_user_acc_stage1, v_user_acc_stage1, t_item_acc_stage1, v_item_acc_stage1)
        
        # Plot stage2 loss & acc
        draw_loss_curve_stage2(t_loss_stage2, v_loss_stage2)
        draw_acc_curve_stage2(t_acc_stage2, v_acc_stage2)




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
        "max_review_user" : 20,
        "max_review_item" : 50,
        "emb_dim" : 768,
        "co_attention_emb_dim" : 512,
        "mf_emb_dim" : 128,
        "lda_group_num": 6, # Include default 0 group. 
        "word_cnn_ksize" : 5,   # odd number 
        "sentence_cnn_ksize" : 3,   # odd number 
        "batch_size": 32,
        "collab_learning": True,
        "epoch" : 10, # when "collab_learning" is False
        "epoch_stage1" : 30, # when "collab_learning" is True
        "epoch_stage2" : 10, # when "collab_learning" is True
        "trade_off_stage1": 0.4, # when "collab_learning" is True, portion of soft-label
        "trade_off_stage2": 0.4, # when "collab_learning" is True, portion of soft-label
    }

    print("Device: ", device)
    print("Collab: ", args["collab_learning"])
    if args["collab_learning"]:
        print("Stage1 Epochs: ", args["epoch_stage1"], "Trade-off: ", args["trade_off_stage1"])
        print("Stage2 Epochs: ", args["epoch_stage2"], "Trade-off: ", args["trade_off_stage2"])
    else:
        print("Epoch: ", args["epoch"])

    main(**args)
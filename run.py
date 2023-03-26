import torch
import torch.nn as nn
import time
import datetime
from function.review_dataset import ReviewDataset, ReviewDataseStage1
from model.hian import HianModel
from model.co_attention import CoattentionNet
from model.fc_layer_stage1 import FcLayerStage1
from model.fc_layer import FcLayer
from model.hian_cl_stage1 import HianCollabStage1
from model.hian_cl_stage2 import HianCollabStage2
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertModel, BertConfig
from function.train import train_model, draw_acc_curve, draw_loss_curve
from function.train_collab import train_collab_model, draw_acc_curve, draw_loss_curve

                                                                   
def main(**args):

    # Dataset/loader
    train_dataset = ReviewDataset(args, target="train")
    val_dataset = ReviewDataset(args, target="val")
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=True)

    train_dataset_stage1 = ReviewDataseStage1(args, target="train")
    val_dataset_stage1 = ReviewDataseStage1(args, target="val")
    train_loader_stage1 = DataLoader(train_dataset_stage1, batch_size=args["batch_size"], shuffle=True)
    val_loader_stage1 = DataLoader(val_dataset_stage1, batch_size=args["batch_size"], shuffle=True)
    

    if not args["collab_learning"]:
        # Init model
        user_network_model = HianModel(args).to(device)
        item_network_model = HianModel(args).to(device)
        co_attention = CoattentionNet(args, args["co_attention_emb_dim"]).to(device)
        fc_layer = FcLayer(args).to(device)

        # Loss criteria
        criterion = nn.CrossEntropyLoss()
        params = list(user_network_model.parameters()) + list(item_network_model.parameters()) + list(co_attention.parameters()) + list(fc_layer.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

        # Training 
        start = time.time()
        train_loss, train_acc, val_loss, val_acc = train_model(args, 
                                                               train_loader, 
                                                               val_loader, 
                                                               user_network_model, 
                                                               item_network_model, 
                                                               co_attention, 
                                                               fc_layer, 
                                                               criterion=criterion, 
                                                               models_params=params, 
                                                               optimizer=optimizer)
        end = time.time()

        print("模型總訓練時間：", str(datetime.timedelta(seconds=int(end - start))))
        draw_loss_curve(train_loss, val_loss)
        draw_acc_curve(train_acc, val_acc)

    elif args["collab_learning"]:
        # Init model
        # Stage 1
        user_network_stage1 = HianCollabStage1(args).to(device)
        item_network_stage1 = HianCollabStage1(args).to(device)
        user_fc_layer_stage1 = FcLayerStage1(args).to(device)
        user_fc_layer_1_stage1 = FcLayerStage1(args).to(device)
        user_fc_layer_2_stage1 = FcLayerStage1(args).to(device)
        user_fc_layer_3_stage1 = FcLayerStage1(args).to(device)
        item_fc_layer_stage1 = FcLayerStage1(args).to(device)
        item_fc_layer_1_stage1 = FcLayerStage1(args).to(device)
        item_fc_layer_2_stage1 = FcLayerStage1(args).to(device)
        item_fc_layer_3_stage1 = FcLayerStage1(args).to(device)

        #Stage2
        user_network_stage2 = HianCollabStage2(args).to(device)
        item_network_stage2 = HianCollabStage2(args).to(device)
        co_attention = CoattentionNet(args, args["co_attention_emb_dim"]).to(device)
        co_attention_1 = CoattentionNet(args, args["co_attention_emb_dim"]).to(device)
        co_attention_2 = CoattentionNet(args, args["co_attention_emb_dim"]).to(device)
        co_attention_3 = CoattentionNet(args, args["co_attention_emb_dim"]).to(device)
        fc_layer_stage2 = FcLayer(args).to(device)
        fc_layer_1_stage2 = FcLayer(args).to(device)
        fc_layer_2_stage2 = FcLayer(args).to(device)
        fc_layer_3_stage2 = FcLayer(args).to(device)

        # Loss criteria
        user_criterion_stage1 = nn.CrossEntropyLoss()
        item_criterion_stage1 = nn.CrossEntropyLoss()
        criterion_stage2 = nn.CrossEntropyLoss()

        # Parameters
        user_params_stage1 = (list(user_network_stage1.parameters()) + 
                              list(user_fc_layer_stage1.parameters()) +
                              list(user_fc_layer_1_stage1.parameters()) +
                              list(user_fc_layer_2_stage1.parameters()) +
                              list(user_fc_layer_3_stage1.parameters()))
        item_params_stage1 = (list(item_network_stage1.parameters()) +
                              list(item_fc_layer_stage1.parameters()) +
                              list(item_fc_layer_1_stage1.parameters()) +
                              list(item_fc_layer_2_stage1.parameters()) +
                              list(item_fc_layer_3_stage1.parameters()))  
        params_stage2 = (list(user_network_stage2.parameters()) + 
                         list(item_network_stage2.parameters()) +
                         list(co_attention.parameters()) + 
                         list(co_attention_1.parameters()) + 
                         list(co_attention_2.parameters()) + 
                         list(co_attention_3.parameters()) + 
                         list(fc_layer_stage2.parameters()) + 
                         list(fc_layer_1_stage2.parameters()) +
                         list(fc_layer_2_stage2.parameters()) + 
                         list(fc_layer_3_stage2.parameters())
                         )
        
        user_optimizer_stage1 = torch.optim.Adam(user_params_stage1, lr=1e-3, weight_decay=1e-4)
        item_optimizer_stage1 =  torch.optim.Adam(item_params_stage1, lr=1e-3, weight_decay=1e-4)
        optimizer_stage2 = torch.optim.Adam(params_stage2, lr=1e-3, weight_decay=1e-4)

        # Training 
        start = time.time()
        train_loss, train_acc, val_loss, val_acc = train_collab_model(
            args,                                                           
            [train_loader_stage1, train_loader],
            [val_loader_stage1, val_loader],
            [user_network_stage1, user_network_stage2],
            [item_network_stage1, item_network_stage2], 
            [co_attention, co_attention_1, co_attention_2, co_attention_3], 
            [user_fc_layer_stage1, user_fc_layer_1_stage1, user_fc_layer_2_stage1, user_fc_layer_3_stage1],
            [item_fc_layer_stage1, item_fc_layer_1_stage1, item_fc_layer_2_stage1, item_fc_layer_3_stage1],
            [fc_layer_stage2, fc_layer_1_stage2, fc_layer_2_stage2, fc_layer_3_stage2], 
            criterions=[user_criterion_stage1, item_criterion_stage1, criterion_stage2], 
            models_params=[user_params_stage1, item_params_stage1, params_stage2], 
            optimizers=[user_optimizer_stage1, item_optimizer_stage1, optimizer_stage2])
        end = time.time()

    
if __name__ == "__main__":

    #Check CUDA 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = {
        "device" : device,
        "train_data_dir" : r'../data/train_df.pkl',
        "val_data_dir" : r'../data/val_df.pkl',
        "user_data_dir" : r'../data/user_emb/',
        "item_data_dir" : r'../data/item_emb/',
        "user_mf_data_dir" : r'../data/train_user_mf_emb.pkl',
        "item_mf_data_dir" : r'../data/train_item_mf_emb.pkl',
        "max_word" : 25,
        "max_sentence" : 10,
        "max_review_user" : 10,
        "max_review_item" : 50,
        "emb_dim" : 768,
        "co_attention_emb_dim" : 256,
        "mf_emb_dim" : 128,
        "lda_group_num": 6, # Include default 0 group. 
        "word_cnn_ksize" : 3,   # odd number 
        "sentence_cnn_ksize" : 3,   # odd number 
        "epoch" : 20,
        "stage1_epoch" : 20,
        "stage2_epoch" : 20,
        "batch_size": 32,
        "collab_learning": True,
        "trade_off": 0.5, 
        "bert_configuration" : BertConfig(),
        "bert_model" : BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(device),
        "bert_tokenizer" : BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    }

    print("Device: ", device)
    main(**args)


import torch
import torch.nn as nn
import time
from function.review_dataset import ReviewDataset   
from model.hian_base import HianModel
from model.co_attention import CoattentionNet
from model.fc_layer import FcLayer
from train import train_model, draw_acc_curve, draw_loss_curve
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertModel, BertConfig

                                                                   
def main(**args):

    # Dataset/loader
    train_dataset = ReviewDataset(args, target="train")
    val_dataset = ReviewDataset(args, target="val")
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=True)

    # Init model
    user_network_model = HianModel(args).to(device)
    item_network_model = HianModel(args).to(device)
    co_attention_network = CoattentionNet(args, args["co_attention_emb_dim"]).to(device)
    fc_layer = FcLayer(args).to(device)

    # Loss criteria
    criterion = nn.MSELoss()
    params = list(user_network_model.parameters()) + list(item_network_model.parameters()) + list(co_attention_network.parameters()) + list(fc_layer.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0005, weight_decay=1e-5)

    # Training 
    start = time.time()
    train_loss, train_acc, val_loss, val_acc = train_model(args, train_loader, val_loader, user_network_model, item_network_model, co_attention_network, fc_layer,
                 criterion=criterion, models_params=params, optimizer=optimizer)
    end = time.time()
    print("模型總訓練時間：%f 秒" % (end - start))
    draw_loss_curve(train_loss, val_loss)
    draw_acc_curve(train_acc, val_acc)
    


if __name__ == "__main__":

    #Check CUDA 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = {
        "device" : device,
        "train_data_dir" : r'../data/train_df.pkl',
        "val_data_dir" : r'../data/val_df.pkl',
        "user_mf_data_dir" : r'../data/train_user_mf_emb.pkl',
        "item_mf_data_dir" : r'../data/train_item_mf_emb.pkl',
        "user_data_dir" : r'../data/user_emb/',
        "item_data_dir" : r'../data/item_emb/',
        "data_chunks_dir" : r'../data/chunks/',
        "emb_dim" : 768,
        "co_attention_emb_dim" : 256,
        "max_word" : 25,
        "max_sentence" : 10,
        "max_review_user" : 10,
        "max_review_item" : 50,
        "lda_group_num": 6, #Include default 0 group. 
        "word_cnn_ksize" : 3,   #odd number 
        "sentence_cnn_ksize" : 3,   #odd number 
        "epoch" : 20,
        "batch_size": 32,
        "bert_configuration" : BertConfig(),
        "bert_model" : BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(device),
        "bert_tokenizer" : BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    }

    print("Device: ",device)
    main(**args)


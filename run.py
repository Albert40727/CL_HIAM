import torch
from function.review_dataset import ReviewDataset   
from model.hian_base import HianModel
from model.co_attention import CoattentionNet
from train import train_model
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertModel, BertConfig

                                                                   
def main(**args):
    dataset = ReviewDataset(args)
    train_loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)
    user_network_model = HianModel(args).to(device)
    item_network_model = HianModel(args).to(device)
    co_attention_network = CoattentionNet(args, args["co_attention_emb_dim"]).to(device)
    train_model(args, train_loader, user_network_model, item_network_model, co_attention_network)

if __name__ == "__main__":

    #Check CUDA 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = {
        "device" : device,
        "data_dir" : r'../data/filtered_reviews_group.pkl',
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
        "epoch" : 5,
        "batch_size": 32,
        "bert_configuration" : BertConfig(),
        "bert_model" : BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(device),
        "bert_tokenizer" : BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True),
    }

    print("Device: ",device)
    main(**args)


import torch
import os 
from function.custom_dataset import ReviewDataset   
from model.hian_base import HianModel
from tqdm import tqdm
from train import train_model
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertModel, BertConfig
import torch.utils.data as data

                                                                   
def main(**args):

    #Get .h5 file chunks and concat them into ConcatDataset
    list_of_datasets = []
    for h5 in os.listdir(args["data_chunks_dir"]):
        if not h5.endswith('.h5'):
            continue  # skip non-h5 files
        list_of_datasets.append(ReviewDataset(h5, args))
    concat_dataset = data.ConcatDataset(list_of_datasets)

    # dataset = ReviewDataset(args)
    train_loader = DataLoader(concat_dataset, batch_size=args["batch_size"], shuffle=False)
    model = HianModel(args).to(device)
    train_model(concat_dataset, train_loader, model, args)

if __name__ == "__main__":

    #Check CUDA 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = {
        "device" : device,
        "data_dir" : r'../data/filtered_reviews_0-999.h5',
        "data_chunks_dir" : r'../data/chunks',
        "emb_dim" : 768,
        "max_word" : 25,
        "max_sentence" : 10,
        "max_review_user" : 10,
        "max_review_item" : 30,
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


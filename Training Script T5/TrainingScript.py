# -*- coding: utf-8 -*-
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import random
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import os
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import nltk


if __name__ == "__main__":
    
    # Command-line Arguments
    dataTrainPath = sys.argv[1]
    dataTestPath = sys.argv[2]
    savePath = sys.argv[3]
    #-------------------------------
    
    
    # DATASET
    class CustomDataset(Dataset):
        def __init__(self, df, tokenizer):
            self.data = df
            #.set_index('index').T.to_dict('list')
            self.tokenizer = tokenizer
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            # Extract the values as strings. They are Pandas
            src_method = self.data['src_method'][idx]
            src_doc = self.data['src_javadoc'][idx]
            dst_method = self.data['dst_method'][idx]
            dst_doc = self.data['dst_javadoc'][idx]
    
            # Tokenize
            srcMTokens = self.tokenizer(src_method, return_tensors="pt", padding='max_length', truncation=True).input_ids
            dstMTokens = self.tokenizer(dst_method, return_tensors="pt", padding='max_length', truncation=True).input_ids
            srcDTokens = self.tokenizer(src_doc, return_tensors="pt", padding='max_length', truncation=True).input_ids
            dstDTokens = self.tokenizer(dst_doc, return_tensors="pt", padding='max_length', truncation=True).input_ids
    
            input_ids = torch.cat([srcMTokens, dstMTokens, srcDTokens], dim=1)
            labels = dstDTokens
    
            # Return the method and line_links as a tuple
            return input_ids, labels
    
    
    # Training Func
    def train_my_model(train_loader,val_loader,num_epochs,model,tokenizer,criterion,optimizer,target_folder, name_experiment, with_validation=False):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      print("Device: {}".format(device))
      model.to(device)
    
      total_steps = len(train_loader)
      t1 = time.time()
    
      training_losses, training_accuracies = [], []
      validation_losses, validation_accuracies = [], []
      model.train()
      for epoch in range(num_epochs):
          training_loss, training_acc = [], []
          validation_loss, validation_acc = [], []
    
          model.train()
          for i, data in enumerate(tqdm(train_loader)):
              inputs, labels = data[0].to(device), data[1].to(device)
              inputs = inputs.squeeze(dim=1)
              labels = labels.squeeze(dim=1)
              # Forward pass
              outputs = model(input_ids=inputs, labels=labels)
              loss = outputs[0]
              # Backprop and optimization
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              training_loss.append(loss.detach().cpu())
          training_losses.append(np.mean(training_loss))
    
          if with_validation and epoch%5==0:
            model.eval()
            print("--- validation")
            for i, data in enumerate(tqdm(val_loader)):
              inputs, labels = data[0].to(device), data[1].to(device)
              inputs = inputs.squeeze(dim=1)
              labels = labels.squeeze(dim=1)
              outputs = model(input_ids=inputs, labels=labels)
              loss = outputs[0]
              validation_loss.append(loss.detach().cpu())
            validation_losses.append(np.mean(validation_loss))
          print('Epoch [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, loss.item()))
    
    
      if with_validation:
        dict = {'Training loss': training_losses}
        dict_val = {'Validation loss': validation_losses}
        df = pd.DataFrame(dict_val)
        # saving the dataframe
        df.to_csv(os.path.join(target_folder,name_experiment+'-validation.csv'))
      else:
        dict = {'Training loss': training_losses}
        df = pd.DataFrame(dict)
        # saving the dataframe
        df.to_csv(os.path.join(target_folder,name_experiment+'-train.csv'))
      print("Saving the model in ",target_folder)
      torch.save(model.state_dict(), os.path.join(target_folder,'model-'+name_experiment+'.ckpt'))
      print("######## Training Finished in {} seconds ###########".format(time.time()-t1))
      
      
      # Run Exp Func
    def run_exp(model,tokenizer,config,trainData, testData, saveFolder):
          print("running experiment ",config['name'],"\n")
        
          #loading the data
          df = pd.read_csv(trainData)
          # Manipulating the dataframe
          df = df.dropna()
          df.reset_index(drop=True, inplace=True)
          
          #loading test
          dfT = pd.read_csv(testData)
          dfT = dfT.dropna()
          dfT.reset_index(drop=True, inplace=True)
        
          #Creating dataset and loaders
          datasetTrain = CustomDataset(df, tokenizer)
          datasetTest = CustomDataset(dfT, tokenizer)
        
          #training parameters
          learning_rate = config['lr']
          batch_size = config['batch_size']
        
          #criterion and optimizer
          criterion = nn.CrossEntropyLoss()
          optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        
          #dataloader
          train_loader = torch.utils.data.DataLoader(datasetTrain, batch_size=batch_size, shuffle=True)
          val_loader = torch.utils.data.DataLoader(datasetTest, batch_size=batch_size, shuffle=True)
        
          #setting up the training and folders were experiments will be saved
          target_folder = saveFolder
          name_experiment = config['name']
          val = True
          #training
          epochs = config['num_epochs']
          train_my_model(train_loader,val_loader,epochs,model,tokenizer,criterion,optimizer,target_folder, name_experiment, with_validation=True)
        
          print("Training for experiment ",config['name']," is done!")
    
    
    # Running
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
    
    config = {'name': 'CodeT5-25epoch-SRC-DST-SRC', 'batch_size': 8, 'num_epochs': 25, 'lr': 0.001}

    run_exp(model,tokenizer,config,dataTrainPath,dataTestPath, savePath)
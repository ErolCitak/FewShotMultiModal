from itertools import cycle
import os
from sched import scheduler
from datetime import datetime
from cv2 import mean
import numpy as np
import tensorboard
from tqdm import tqdm
import pandas as pd

from dataloader_2d import HVUDataset
import vae_2d
import torch
from torch.nn.functional import one_hot
from torch import optim

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

def initialize_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_dataset(folder_path, n_way=3, k_shot=5, n_test_per_class=1, multimodal=True, transform=None):
    # initiate dataset class
    dataset = HVUDataset(folder_path, n_way, k_shot, n_test_per_class, multimodal, transform)
    
    return dataset

def init_optim(init_lr, model):
    # initiate optimizer
    
    # Adam
    
    optim = torch.optim.Adam(params=model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    
    # SGD
    #optim = torch.optim.SGD(params=model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0005)
    
    return optim

def init_lr_scheduler(optimizer, base_lr=0.001, max_lr=0.1, n_iter=6500, n_up=2, n_down=2, n_epoch=30):
    # initiate optimizer scheduler
    # naive adam
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90], gamma=0.5)
    
    # naive sgd
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=n_iter*n_up, step_size_down=n_iter*n_down, mode='triangular')
    
    # cycle adam
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=n_iter, epochs=n_epoch)
    
    return scheduler

def init_model(latent_visual=128, latent_textual=128, multimodal=True, kl_coef=0.5):
    # initiate vae model
    model = vae_2d.vae(latent_visual=latent_visual, latent_textual=latent_textual, multimodal=multimodal, kl_coef=kl_coef)
    
    return model
    
def save_list_to_file(path, thelist):
    with open(path, 'a') as f:
        if type(thelist) is not list:
            f.write("%s\n" % thelist)
        else:
            for item in thelist:
                f.write("%s\n" % item)
                
def loss_annealing(current_epoch):
        
        reconstruction_factor = torch.cuda.FloatTensor([1.0])
        
        f1 = 1.0*(current_epoch - 20)/(1.0*(75 - 20))
        f1 = f1*(1.0* 1.5)
        cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1,0),  1.5)])

        f2 = 1.0 * (current_epoch - 0) / ( 1.0 * (90 - 0))
        f2 = f2 * (1.0 * 0.25)
        kl_loss_factor = torch.cuda.FloatTensor([min(max(f2, 0), 0.25)])

        f3 = 1.0*(current_epoch - 10)/(1.0*(30 - 10))
        f3 = f3*(1.0* 3)
        distance_factor = torch.cuda.FloatTensor([min(max(f3,0), 3)])

        
        f4 = 1.0*(current_epoch - 30)/(1.0*(90 - 30))
        f4 = f4*(1.0* 0.25)   
        #ce_factor = torch.cuda.FloatTensor([min(max(f4,0), 0.25)])
        ce_factor = torch.cuda.FloatTensor([0.0])
                        
        return reconstruction_factor, cross_reconstruction_factor, kl_loss_factor, distance_factor, ce_factor
         
def train(tr_dataloader, model, optim, lr_scheduler, 
          n_way=None, k_shot=None, n_epoch=None, n_iter= None, n_iter_val = None,
          val_dataloader=None, save_model_path=None, multimodal=None, hidden_size=None, k_shot_test=None, log_file=None,
          recons_coef = 1.0, kl_loss_coeff = 1.0, kl_beta = 0.5, cross_recons_coeff = 0.2, dist_align_coeff = 0.1, dist_aware_ce_coeff = 0.1,
          tensorboard_writer=None, freeze_decoder=-1):

    # train the model with the prototypical learning algorithm

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if val_dataloader is None:
        best_state = None
    
    # recons_coef, cross_recons_coeff, kl_loss_coeff, dist_align_coeff, dist_aware_ce_coeff
    loss_coefficents = {"recons_coef":recons_coef, "cross_recons_coeff":cross_recons_coeff, "kl_loss_coeff":kl_loss_coeff, "dist_align_coeff":dist_align_coeff, "dist_aware_ce_coeff":dist_aware_ce_coeff}
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # for monitoring purposes
    cons_loss_list = []
    recons_loss_list = []
    kl_loss_list = []
    dist_loss_list = []
    ce_loss_list = []
    margin_loss_list = []
    
    best_loss = np.inf
    best_model_path = os.path.join(save_model_path, 'best_model_')
    
    print("=== Device: {} ===".format(device), file=log_file)
    print('=== N-way: {} -- K-shot: {} ==='.format(n_way, k_shot), file=log_file)
    for epoch in range(n_epoch):
        
        print('\n=== Epoch: {}/{} ==='.format(epoch, n_epoch), file=log_file)
        print('=== Learning Rate: {} ==='.format(lr_scheduler.get_last_lr()), file=log_file)
        
        recons_coef, cross_recons_coeff, kl_loss_coeff, dist_align_coeff, dist_aware_ce_coeff = loss_annealing(epoch)
        loss_coefficents = {"RC":recons_coef, "C-RC":cross_recons_coeff, "KL":kl_loss_coeff, "Dist-Algn":dist_align_coeff, "CE":dist_aware_ce_coeff}
        print(loss_coefficents, file=log_file)
        
        if epoch >= 25:
            best_loss = np.inf
            
        if (epoch >= freeze_decoder) and (freeze_decoder != -1):
            # freeze the decoder
            for name, param in model.named_parameters():
                if (param.requires_grad) and ("decoder" in name):
                    param.requires_grad = False
                    
                    # and set the coefficient of decoder related 
                    # functions
                    recons_coef = 0.0
                    cross_recons_coeff = 0.0 
                    kl_loss_coeff = 0.0 
                    dist_align_coeff = 0.0                  
        
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("# of trainable params: {}".format(params))            
        model.train()
        
        
        # for monitoring purposes
        iter_cons_loss_list = []
        iter_recons_loss_list = []
        iter_kl_loss_list = []
        iter_dist_loss_list = []
        iter_ce_loss_list = []
        iter_margin_loss_list = []
        
        for iter in tqdm(range(n_iter)): 
            optim.zero_grad()
            
            loss = 0
            # get the data batch
            # e.g., 3-way k-shot problem; len(x_visual) = 3
            # then each element of x_visual has 6 samples, 5 of them training, 1 of them testing
            if multimodal:
                #x_visual, x_textual  = tr_dataloader.next_batch()            
                dataloader_result = tr_dataloader.next_batch()  
                x_visual, y_visual, x_test_visual, y_test_visual, x_textual, y_textual, x_test_textual, y_test_textual \
                    = dataloader_result["visual_data_train"], dataloader_result["visual_data_train_label"] \
                    , dataloader_result["visual_data_test"], dataloader_result["visual_data_test_label"] \
                    , dataloader_result["textual_data_train"], dataloader_result["textual_data_train_label"] \
                    , dataloader_result["textual_data_test"], dataloader_result["textual_data_test_label"]
                
                # for batch wise processing
                x_visual = torch.stack(x_visual)
                if len(x_test_visual) > 0:
                    x_test_visual = torch.stack(x_test_visual)
                x_visual = torch.cat((x_visual, x_test_visual),0).to(device)
                    
                x_textual = torch.stack(x_textual)
                if len(x_test_textual) > 0:
                    x_test_textual = torch.stack(x_test_textual)
                x_textual = torch.cat((x_textual, x_test_textual),0).to(device)
                
                y_visual = torch.tensor(y_visual).long()
                if len(y_test_visual) > 0:
                    y_test_visual = torch.tensor(y_test_visual).long()
                y_visual = torch.cat((y_visual, y_test_visual),0).to(device)
                    
                y_textual = torch.tensor(y_textual).long()
                if len(y_test_textual) > 0:
                    y_test_textual = torch.tensor(y_test_textual).long()
                y_textual = torch.cat((y_textual, y_test_textual),0).to(device)
                
                ######                            
                # feed the all inputs to the model
                ######
                model_result = model(x_visual, x_textual)
                
                ######
                # calculate the loss scores
                ######
                
                # reconstruction losses
                visual_from_visual_recons_loss = model.reconstruction_loss(x_visual[:n_way*k_shot], model_result["visual_decoded_output_from_visual"][:n_way*k_shot])
                visual_from_textual_recons_loss = model.reconstruction_loss(x_visual[:n_way*k_shot], model_result["visual_decoded_output_from_textual"][:n_way*k_shot])
                
                textual_from_textual_recons_loss = model.reconstruction_loss(x_textual[:n_way*k_shot], model_result["textual_decoded_output_from_textual"][:n_way*k_shot])
                textual_from_visual_recons_loss = model.reconstruction_loss(x_textual[:n_way*k_shot], model_result["textual_decoded_output_from_visual"][:n_way*k_shot])
                
                # kl-divergence
                kl_loss_visual = model.KL_loss(model_result["mu_visual"][:n_way*k_shot], model_result["logvar_visual"][:n_way*k_shot], kl_beta)
                kl_loss_textual = model.KL_loss(model_result["mu_text"][:n_way*k_shot], model_result["logvar_text"][:n_way*k_shot], kl_beta)
                
                # distribution alignment
                dist_align_loss = model.distribution_alignment_loss(model_result["mu_visual"][:n_way*k_shot], model_result["mu_text"][:n_way*k_shot], 
                                                                    model_result["logvar_visual"][:n_way*k_shot], model_result["logvar_text"][:n_way*k_shot])   
                
                # dist. aware classification loss
                dist_aware_result = model.dist_aware_classification(mus=model_result["mu_visual"][:n_way*k_shot], logvars=model_result["logvar_visual"][:n_way*k_shot], z_train = model_result["z_visual"][:n_way*k_shot],
                                                                    test_mus=model_result["mu_visual"][n_way*k_shot:], test_logvars=model_result["logvar_visual"][n_way*k_shot:], z_test = model_result["z_visual"][n_way*k_shot:], 
                                                                    n_way=n_way, k_shot=k_shot, hidden_size=hidden_size, k_shot_test=k_shot_test, gt_tensor=y_visual[n_way*k_shot:])
                
                dist_ce_loss, dist_margin_loss, predicted_labels = dist_aware_result["cross_entropy_loss"], dist_aware_result["margin_loss"], dist_aware_result["predicted_labels"]
                
                
                # Own monitor app
                if (iter % 299 == 0) and (iter != 0): 
                    print("###################\nLOSSES TRAIN- Epoch: {} Iter: {}\n###################".format(epoch, iter), file=log_file)
                    print("Recons_v_v: {:.5f}".format(visual_from_visual_recons_loss), file=log_file)
                    print("Recons_t_t: {:.5f}".format(textual_from_textual_recons_loss), file=log_file)
                    print("Cross_Recons_v_t: {:.5f}".format(visual_from_textual_recons_loss), file=log_file)
                    print("Cross_Recons_t_v: {:.5f}".format(textual_from_visual_recons_loss), file=log_file)
                    print("KL_v: {:.5f}".format(kl_loss_visual), file=log_file)
                    print("KL_t: {:.5f}".format(kl_loss_textual), file=log_file)
                    print("Dist_Align: {:.5f}".format(dist_align_loss), file=log_file)
                    print("CE: {:.5f}".format(dist_ce_loss.item()), file=log_file)
                    print("Pred_Labels: {}".format(predicted_labels), file=log_file)

                
                # Ref: https://github.com/edgarschnfld/CADA-VAE-PyTorch/blob/26f0085fe5e5911dc06fe767f90965c47885dee1/model/vaemodel.py#L131
                loss = (recons_coef * (visual_from_visual_recons_loss + textual_from_textual_recons_loss)) + (kl_loss_coeff * (kl_loss_visual + kl_loss_textual)) 
                
                # for monitoring purposes
                iter_cons_loss_list.append((visual_from_visual_recons_loss + textual_from_textual_recons_loss).item())
                iter_recons_loss_list.append((visual_from_textual_recons_loss + textual_from_visual_recons_loss).item())
                iter_kl_loss_list.append((kl_loss_visual + kl_loss_textual).item())
                iter_dist_loss_list.append(dist_align_loss.item())
                iter_ce_loss_list.append(dist_ce_loss.item())
                iter_margin_loss_list.append(dist_margin_loss.item())
                
                
                if cross_recons_coeff > 0:
                    loss += (cross_recons_coeff * (visual_from_textual_recons_loss + textual_from_visual_recons_loss)) 
                if dist_align_coeff > 0:
                    loss += dist_align_coeff * dist_align_loss 
                    loss += dist_align_coeff * dist_margin_loss
                if dist_aware_ce_coeff > 0:
                    loss += dist_aware_ce_coeff * dist_ce_loss
                    
                    
                    
                #print("Total Loss: {}".format(loss))
            
            else: # TO.DO! For unimodal
                x_visual = tr_dataloader.next_batch()
                x_visual = x_visual.to(device)
            
            #print("Loss: {:.5f}".format(loss))         
            loss.backward()
            optim.step()
            # For cyclic and one cycle
            #lr_scheduler.step()
            train_loss.append(loss.item())
        ## Iteration Ends, eg 6500 iteration
        
        # for monitoring purposes, when the epoch ends save them into a another txt file
        # for each iteration result
        cons_loss_list.append(iter_cons_loss_list)
        recons_loss_list.append(iter_recons_loss_list)
        kl_loss_list.append(iter_kl_loss_list)
        dist_loss_list.append(iter_dist_loss_list)
        ce_loss_list.append(iter_ce_loss_list)
        margin_loss_list.append(iter_margin_loss_list)
        
        # Tensorboard Log
        tensorboard_writer.add_scalar("2-Recons_Loss", sum(iter_cons_loss_list) / len(iter_cons_loss_list), epoch)
        tensorboard_writer.add_scalar("3-Cross-Recons_Loss", sum(iter_recons_loss_list) / len(iter_recons_loss_list), epoch)
        tensorboard_writer.add_scalar("4-KL_Loss", sum(iter_kl_loss_list) / len(iter_kl_loss_list), epoch)
        tensorboard_writer.add_scalar("5-Dist_Align_Loss", sum(iter_dist_loss_list) / len(iter_dist_loss_list), epoch)
        tensorboard_writer.add_scalar("0-CE_Loss", sum(iter_ce_loss_list) / len(iter_ce_loss_list), epoch)
        tensorboard_writer.add_scalar("00-Margin_Loss", sum(iter_margin_loss_list) / len(iter_margin_loss_list), epoch)
        tensorboard_writer.add_scalar("6-Train_Loss",  np.mean(train_loss), epoch)
                
        # For exponential 
        lr_scheduler.step()
        avg_loss_train = np.mean(train_loss)
        #avg_acc = np.mean(train_acc[-n_iter:])
        #print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        print('Avg Train Loss: {}'.format(avg_loss_train), file=log_file)
        
        
        # Val performance
        if val_dataloader is None:
            continue
        else:
            val_loss, val_acc = validate(model, val_dataloader, n_iter_val, hidden_size, kl_loss_coeff, kl_beta, recons_coef, cross_recons_coeff, dist_align_coeff, dist_aware_ce_coeff, device,k_shot_test)                  
            #val_acc.append(acc.item())

            avg_loss_val = np.mean(val_loss)
            avg_val_acc = np.mean(val_acc)
            
            tensorboard_writer.add_scalar("7-Val_Loss", avg_loss_val.item(), epoch)
            tensorboard_writer.add_scalar("1-Val_Acc", avg_val_acc.item(), epoch)
            
            print('Avg Val Loss: {}, Previously Best Loss: {}'.format(avg_loss_val, best_loss), file=log_file)
            if (avg_loss_val < best_loss):
                torch.save(model.state_dict(), best_model_path+str(avg_loss_val)+'_'+str(epoch)+'-epoch.pth')
                best_loss = avg_loss_val
                best_state = model.state_dict()

            for name in ['avg_loss_train', 'avg_loss_val']:
                save_list_to_file(os.path.join(save_model_path, name + '.txt'), locals()[name])
                
            print("------------------------------------------------------------"+str(epoch)+" finished------------------------------------------------------------")

    # when the all operation ends
    # save each loss seperately
    df = pd.DataFrame(data=zip(cons_loss_list,recons_loss_list,kl_loss_list,dist_loss_list, ce_loss_list) , 
                 columns=['cons_loss','recons_loss','kl_loss','dist_loss','ce_loss'])
    
    return best_state, train_loss, val_loss, df

def validate(model, data_loader, n_iter, hidden_size, kl_loss_coeff, kl_beta, recons_coef, cross_recons_coeff, dist_align_coeff, dist_aware_ce_coeff, device, k_shot_test,
             tensorboard_writer=None):
    val_loss = []
    val_acc = []
    model.eval()
    
    with torch.no_grad():
        for iter in tqdm(range(n_iter)):    
            loss = 0
            # get the data batch
            # e.g., 3-way k-shot problem; len(x_visual) = 3
            # then each element of x_visual has 6 samples, 5 of them training, 1 of them testing
            if multimodal:
                #x_visual, x_textual  = tr_dataloader.next_batch()            
                dataloader_result = data_loader.next_batch()  
                x_visual, y_visual, x_test_visual, y_test_visual, x_textual, y_textual, x_test_textual, y_test_textual \
                    = dataloader_result["visual_data_train"], dataloader_result["visual_data_train_label"] \
                    , dataloader_result["visual_data_test"], dataloader_result["visual_data_test_label"] \
                    , dataloader_result["textual_data_train"], dataloader_result["textual_data_train_label"] \
                    , dataloader_result["textual_data_test"], dataloader_result["textual_data_test_label"]
                
                # for batch wise processing
                x_visual = torch.stack(x_visual)
                if len(x_test_visual) > 0:
                    x_test_visual = torch.stack(x_test_visual)
                x_visual = torch.cat((x_visual, x_test_visual),0).to(device)
                    
                x_textual = torch.stack(x_textual)
                if len(x_test_textual) > 0:
                    x_test_textual = torch.stack(x_test_textual)
                x_textual = torch.cat((x_textual, x_test_textual),0).to(device)
                
                y_visual = torch.tensor(y_visual).long()
                if len(y_test_visual) > 0:
                    y_test_visual = torch.tensor(y_test_visual).long()
                y_visual = torch.cat((y_visual, y_test_visual),0).to(device)
                    
                y_textual = torch.tensor(y_textual).long()
                if len(y_test_textual) > 0:
                    y_test_textual = torch.tensor(y_test_textual).long()
                y_textual = torch.cat((y_textual, y_test_textual),0).to(device)
                
                ######                            
                # feed the all inputs to the model
                ######
                model_result = model(x_visual, x_textual)
                
                ######
                # calculate the loss scores
                ######
                
                # reconstruction losses
                visual_from_visual_recons_loss = model.reconstruction_loss(x_visual[:n_way*k_shot], model_result["visual_decoded_output_from_visual"][:n_way*k_shot])
                visual_from_textual_recons_loss = model.reconstruction_loss(x_visual[:n_way*k_shot], model_result["visual_decoded_output_from_textual"][:n_way*k_shot])
                
                textual_from_textual_recons_loss = model.reconstruction_loss(x_textual[:n_way*k_shot], model_result["textual_decoded_output_from_textual"][:n_way*k_shot])
                textual_from_visual_recons_loss = model.reconstruction_loss(x_textual[:n_way*k_shot], model_result["textual_decoded_output_from_visual"][:n_way*k_shot])
                
                # kl-divergence
                kl_loss_visual = model.KL_loss(model_result["mu_visual"][:n_way*k_shot], model_result["logvar_visual"][:n_way*k_shot], kl_beta)
                kl_loss_textual = model.KL_loss(model_result["mu_text"][:n_way*k_shot], model_result["logvar_text"][:n_way*k_shot], kl_beta)
                
                # distribution alignment
                dist_align_loss = model.distribution_alignment_loss(model_result["mu_visual"][:n_way*k_shot], model_result["mu_text"][:n_way*k_shot], 
                                                                    model_result["logvar_visual"][:n_way*k_shot], model_result["logvar_text"][:n_way*k_shot])   
                
                # dist. aware classification loss
                dist_aware_result = model.dist_aware_classification(mus=model_result["mu_visual"][:n_way*k_shot], logvars=model_result["logvar_visual"][:n_way*k_shot], z_train = model_result["z_visual"][:n_way*k_shot],
                                                                    test_mus=model_result["mu_visual"][n_way*k_shot:], test_logvars=model_result["logvar_visual"][n_way*k_shot:], z_test = model_result["z_visual"][n_way*k_shot:], 
                                                                    n_way=n_way, k_shot=k_shot, hidden_size=hidden_size, k_shot_test=k_shot_test, gt_tensor=y_visual[n_way*k_shot:])
                
                dist_ce_loss, dist_margin_loss, predicted_labels = dist_aware_result["cross_entropy_loss"], dist_aware_result["margin_loss"], dist_aware_result["predicted_labels"]
                
                gt = y_visual[n_way*k_shot:].cpu().detach().numpy()
                y_hat = predicted_labels.cpu().detach().numpy()

                dist_acc = accuracy_score(gt, y_hat)
                val_acc.append(dist_acc)
                            
                # Ref: https://github.com/edgarschnfld/CADA-VAE-PyTorch/blob/26f0085fe5e5911dc06fe767f90965c47885dee1/model/vaemodel.py#L131
                loss = (recons_coef * (visual_from_visual_recons_loss + textual_from_textual_recons_loss)) + (kl_loss_coeff * (kl_loss_visual + kl_loss_textual)) 
                
                if cross_recons_coeff > 0:
                    loss += (cross_recons_coeff * (visual_from_textual_recons_loss + textual_from_visual_recons_loss)) 
                if dist_align_coeff >0:
                    loss += dist_align_coeff*dist_align_loss 
                if dist_aware_ce_coeff >0:
                    loss += dist_aware_ce_coeff*dist_ce_loss
            
            else: # TO.DO! For unimodal
                x_visual = data_loader.next_batch()
                x_visual = x_visual.to(device)

                # feed the inputs to the model   
                model_output = model(x_visual)

            val_loss.append(loss.item())
        ## Iteration Ends, eg 6500 iteration

    return [val_loss, dist_acc]

def test(model, load_model_path, data_loader, n_way, k_shot, n_iter, hidden_size, k_shot_test, multimodal=True, tensorboard_writer=None,
         visual_based=True, complete_test=False, zero_shot_test=False):
    
    # load the best model
    model.load_state_dict(torch.load(load_model_path))
    model.eval()
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # final acc results
    test_accuracies = []
    
    with torch.no_grad():
        # for each iteration
        for iter in tqdm(range(n_iter)):
            if multimodal:
                #x_visual, x_textual  = tr_dataloader.next_batch()            
                dataloader_result = data_loader.next_batch()  
                x_visual, y_visual, x_test_visual, y_test_visual, x_textual, y_textual, x_test_textual, y_test_textual \
                    = dataloader_result["visual_data_train"], dataloader_result["visual_data_train_label"] \
                    , dataloader_result["visual_data_test"], dataloader_result["visual_data_test_label"] \
                    , dataloader_result["textual_data_train"], dataloader_result["textual_data_train_label"] \
                    , dataloader_result["textual_data_test"], dataloader_result["textual_data_test_label"]
                
                # for batch wise processing
                x_visual = torch.stack(x_visual)
                if len(x_test_visual) > 0:
                    x_test_visual = torch.stack(x_test_visual)
                x_visual = torch.cat((x_visual, x_test_visual),0).to(device)
                    
                x_textual = torch.stack(x_textual)
                if len(x_test_textual) > 0:
                    x_test_textual = torch.stack(x_test_textual)
                x_textual = torch.cat((x_textual, x_test_textual),0).to(device)
                
                y_visual = torch.tensor(y_visual).long()
                if len(y_test_visual) > 0:
                    y_test_visual = torch.tensor(y_test_visual).long()
                y_visual = torch.cat((y_visual, y_test_visual),0).to(device)
                    
                y_textual = torch.tensor(y_textual).long()
                if len(y_test_textual) > 0:
                    y_test_textual = torch.tensor(y_test_textual).long()
                y_textual = torch.cat((y_textual, y_test_textual),0).to(device)
                
                ######                            
                # feed the all inputs to the model
                ######
                model_result = model(x_visual, x_textual)
                
                if zero_shot_test:
                    ## Zero Shot Classification;
                    ### Given an auxiliary info, tries to classify videos 
                    ### based on this info.
                    dist_aware_result = model.dist_aware_classification(mus=model_result["mu_text"][:n_way*k_shot], logvars=model_result["logvar_text"][:n_way*k_shot], z_train = model_result["z_text"][:n_way*k_shot],
                                                                        test_mus=model_result["mu_visual"][n_way*k_shot:], test_logvars=model_result["logvar_visual"][n_way*k_shot:],z_test = model_result["z_visual"][n_way*k_shot:], 
                                                                        n_way=n_way, k_shot=k_shot, hidden_size=hidden_size, k_shot_test=k_shot_test, gt_tensor=y_visual[n_way*k_shot:])
                elif visual_based:
                    ## Visual Based Classification
                    dist_aware_result = model.dist_aware_classification(mus=model_result["mu_visual"][:n_way*k_shot], logvars=model_result["logvar_visual"][:n_way*k_shot], z_train = model_result["z_visual"][:n_way*k_shot],
                                                                        test_mus=model_result["mu_visual"][n_way*k_shot:], test_logvars=model_result["logvar_visual"][n_way*k_shot:], z_test = model_result["z_visual"][n_way*k_shot:], 
                                                                        n_way=n_way, k_shot=k_shot, hidden_size=hidden_size, k_shot_test=k_shot_test, gt_tensor=y_visual[n_way*k_shot:])

                else:
                    ## Textual Based Classification
                    dist_aware_result = model.dist_aware_classification(mus=model_result["mu_text"][:n_way*k_shot], logvars=model_result["logvar_text"][:n_way*k_shot], z_train = model_result["z_text"][:n_way*k_shot],
                                                                        test_mus=model_result["mu_text"][n_way*k_shot:], test_logvars=model_result["logvar_text"][n_way*k_shot:], z_test = model_result["z_text"][n_way*k_shot:], 
                                                                        n_way=n_way, k_shot=k_shot, hidden_size=hidden_size, k_shot_test=k_shot_test, gt_tensor=y_visual[n_way*k_shot:])
                
                dist_ce_loss, dist_margin_loss, predicted_labels = dist_aware_result["cross_entropy_loss"], dist_aware_result["margin_loss"], dist_aware_result["predicted_labels"]
                
                gt = y_visual[n_way*k_shot:].cpu().detach().numpy()
                y_hat = predicted_labels.cpu().detach().numpy()
                
                acc = accuracy_score(gt, y_hat)
                test_accuracies.append(acc)
                        
            else: # TO.DO! For unimodal
                x_visual = data_loader.next_batch()
                x_visual = x_visual.to(device)
    
    avg_acc = np.mean(np.array(test_accuracies))*100
    # if only do once, show everything
    if not complete_test:
        # display the average acc score
        print("Multimodal: {}".format(multimodal))
        print("Zero-Shot: {}".format(do_zero_shot_test))
        print("Visual Based: {}".format(visual_based))
        print("n-way: {} -- k-shot: {}".format(n_way, k_shot))
        print("Average Acc: {:.3f}%".format(avg_acc))
    
    return avg_acc
        
    
if __name__ == "__main__":
    
    # set seed elems
    initialize_seed(42)
    
    n_way = 3 # number of class in one meta-learning setup
    k_shot = 5 # number of samples per class in one meta-learning setup
    n_test_per_class = 15 # how many test samples per class in one meta-learning setup
    n_epoch = 100 # number of epoch for all training
    n_iter = 300 # how many episodes will be used in just one epoch in training
    n_iter_val = 300 #  how many episodes will be used in just one epoch in validation
    n_iter_test = 300 #  how many episodes will be used in just one epoch in test
    base_lr = 0.00001 # base learning rate of cycle learning rate
    init_lr = 0.001 # 0.00001, maximum learning rate of cycle learning rate --> all trials: 0.001
    n_lr_up = 2 # it means, from base_lr -> init_lr takes n_lr_up epoch
    n_lr_down = 2 # it means, from init_lr -> base_lr takes n_lr_down epoch
    latent_size = 64
    multimodal = True # multimodal or unimodal
    freeze_decoder = -1 # -1 --> no freeze otherwise indicates the starting epoch to freezing
    save_model_path = "C:/Users/PC/Desktop/FewShotPhd/model_hvu/vae_2d"
    
    ##########################################################################################
    ########## TEST CONFIGURATIONS ##########
    training_phase = True
    trained_model_name = "14_08_2022_16_20_08_without-ce[base-solution,lr0.001-40,80x0.5]_with-ml[da-coeff]"
    trained_model_epcoch = "best_model_0.4562857644756635_69-epoch.pth"
    trained_few_shot_model = trained_model_name+"/"+trained_model_epcoch
    #trained_few_shot_model = "23_07_2022_16_19_36/best_mode0l_0.11340323341389497_4-epoch.pth"
    do_complete_test = False # it refers to for all N numbers and K numbers --> 3,1 - 3,5 - 3,10 - 3,20 etc and for 10 times
    n_complete_test = 10
    ######
    do_zero_shot_test = False    
    visual_based_test = True # True: video-video classification, False: textual-textual classification !! do_zero_shot overwrites this parameter !!
    ######
    ##########################################################################################
    
    # train / test initializers
    if training_phase:
        # get the beginning of the operation 
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")

        today = datetime.today()
        current_day = today.strftime("%d_%m_%Y_")
        
        
        #specific_name = "_without-ce[base-solution]"
        specific_name = "_without-ce[base-solution,lr0.001-40,80x0.5]_with-ml[da-coeff]"
        current_time = current_day+current_time + specific_name
        

        save_model_path = os.path.join(save_model_path, current_time)
        file_path = os.path.join(save_model_path, "log.txt")

        tensorboard_writer = SummaryWriter(log_dir="C:/Users/PC/Desktop/FewShotPhd/model_hvu/vae_2d/runs/"+current_time)
        
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
            log_file = open(file_path,'a')
    
    else:
        load_model_name = trained_few_shot_model
        save_model_path = os.path.join(save_model_path)
        print(load_model_name.split("_")[-1].split(".")[0])
    
    
    
    # set train, test, val dataloaders
    train_dataloader = init_dataset("C:/HolisticVideoUnderstanding/uniform_train", n_way, k_shot, n_test_per_class=n_test_per_class, multimodal=True, transform=None)
    val_dataloader = init_dataset("C:/HolisticVideoUnderstanding/uniform_val", n_way, k_shot, n_test_per_class=n_test_per_class, multimodal=True, transform=None)
    
    
    # delete txt logs if exists
    if os.path.exists(os.path.join(save_model_path, "avg_loss_val.txt")):
        os.remove(os.path.join(save_model_path, "avg_loss_val.txt"))
    if os.path.exists(os.path.join(save_model_path, "avg_loss_train.txt")):
        os.remove(os.path.join(save_model_path, "avg_loss_train.txt"))
    

    model = init_model(latent_visual=latent_size, latent_textual=latent_size, multimodal=multimodal)
    optim = init_optim(init_lr=init_lr, model=model)
    lr_scheduler = init_lr_scheduler(optimizer=optim, base_lr=base_lr, max_lr=init_lr, n_iter=n_iter, n_up=n_lr_up, n_down=n_lr_down, n_epoch=n_epoch)          
    
    if training_phase:
        res = train(tr_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    model=model, optim=optim, lr_scheduler=lr_scheduler, 
                    n_way=n_way, k_shot=k_shot, n_epoch=n_epoch, n_iter= n_iter, n_iter_val=n_iter_val, hidden_size=latent_size, k_shot_test=n_test_per_class,
                    save_model_path=save_model_path, multimodal=multimodal, log_file = log_file, tensorboard_writer=tensorboard_writer, freeze_decoder=freeze_decoder)
    
        best_state, train_loss, val_loss, df = res
        
        # save df to .csv file 
        df.to_csv(os.path.join(save_model_path, "losses.csv"), index=False)
        
        #print("Best State: {}\nTraining Loss: {}\nVal Loss: {}".format(best_state, train_loss, val_loss))
    
    else:
        load_model_path = os.path.join(save_model_path, load_model_name)
        
        if not do_complete_test:
            
            test_dataloader = init_dataset("C:/HolisticVideoUnderstanding/uniform_test", n_way, k_shot, n_test_per_class=n_test_per_class, 
                                                   multimodal=True, transform=None)
            
            res = test(model= model, load_model_path= load_model_path, data_loader= test_dataloader, n_way= n_way, k_shot= k_shot, 
                    n_iter= n_iter_test, hidden_size= latent_size, k_shot_test=n_test_per_class, multimodal= multimodal, visual_based=visual_based_test,
                    complete_test=do_complete_test, zero_shot_test=do_zero_shot_test)

        else:
            
            # Result will be outputed to txt file
            if do_zero_shot_test:
                name_appendix = "Zero-Shot"
            else:
                name_appendix = "Visual" if visual_based_test else "Textual"
                
            result_file_path = "C:/Users/PC/Desktop/FewShotPhd/result_hvu/"
            result_file_name = trained_few_shot_model.split(".pth")[0].replace("/","--") +"_"+ name_appendix +".txt"
            result_file_full_path = os.path.join(result_file_path, result_file_name)
            
            complete_test_result = []
            
            for current_n_way in [3,5,10]: # number of classes
                for current_k_shot in [1,3,5,10]: # number of training samples per class
                    
                    # we need to re-arrange test dataloader for each time
                    test_dataloader = init_dataset("C:/HolisticVideoUnderstanding/uniform_test", current_n_way, current_k_shot, n_test_per_class=n_test_per_class, 
                                                   multimodal=True, transform=None)
                    
                    print("Current N-Way --> {}".format(current_n_way))
                    print("Current K-Shot --> {}".format(current_k_shot))
                    print("Visual Based Test --> {}".format(visual_based_test))
                    print("Zero-Shot Test: {}".format(do_zero_shot_test))
                    temp_score = []
                    
                    for c_iter in range(n_complete_test):
                        res = test(model= model, load_model_path= load_model_path, data_loader= test_dataloader, n_way= current_n_way, k_shot= current_k_shot, 
                                    n_iter= n_iter_test, hidden_size= latent_size, k_shot_test=n_test_per_class, multimodal= multimodal, visual_based=visual_based_test,
                                    complete_test=do_complete_test, zero_shot_test=do_zero_shot_test)
                        
                        temp_score.append(res)
                    
                    complete_test_result.append({"n_way":current_n_way, "k_shot": current_k_shot, "avg_score": np.mean(np.array(temp_score)), "std_score": np.std(np.array(temp_score))})                    
                    
                    # Write down to the txt file
                    with open(result_file_full_path, 'a') as fp:
                        # write each item on a new line
                        fp.write("N-way: %d\n" % int(current_n_way))
                        fp.write("K-shot: %d\n" % int(current_k_shot))
                        fp.write("Avg Acc: %f\n" % np.mean(np.array(temp_score)))
                        fp.write("Std: %f\n" % np.std(np.array(temp_score)))
                        fp.write("-------------------------------------------------------\n")
                    
            # show the complete test result
            print(complete_test_result)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torchvision
from torch.autograd import Variable
from torchsummary import summary

from sklearn.metrics import classification_report


def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)


    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class vae_visual_encoder(nn.Module):
    def __init__(self, latent_visual=128):
        super(vae_visual_encoder, self).__init__()
        
        self.visual_latent_in = latent_visual

        
        ############
        
        visual_modules = []
        
        # visual encoder
        visual_modules.append(nn.Linear(1536,768))
        visual_modules.append(nn.ReLU())
        #visual_modules.append(nn.Linear(512, self.visual_latent_in*2))
        #visual_modules.append(nn.ReLU())


        
        self.visual_encoder = nn.Sequential(*visual_modules)
        self.fc_mu_visual = nn.Linear(768, latent_visual)
        self.fc_logvar_visual = nn.Linear(768, latent_visual)
        
        self.apply(weights_init)
        
    def forward(self, x_visual):
        
        x_visual = self.visual_encoder(x_visual)
        
        mu_visual = self.fc_mu_visual(x_visual)
        logvar_visual = self.fc_logvar_visual(x_visual)
        

        return mu_visual, logvar_visual
    
#encoder = vae_visual_encoder(latent_visual=128)
#print(summary(encoder,(1,1536), device="cpu"))


class vae_textual_encoder(nn.Module):
    def __init__(self, latent_textual=128):
        super(vae_textual_encoder, self).__init__()
        
        
        self.textual_latent_in = latent_textual

        textual_modules = []
        
        # textual encoder
        textual_modules.append(nn.Linear(1024, 512))
        textual_modules.append(nn.ReLU())
        #textual_modules.append(nn.Linear(512,self.textual_latent_in*2))
        #textual_modules.append(nn.ReLU())
            
        self.textual_encoder = nn.Sequential(*textual_modules)
        self.fc_mu_textual = nn.Linear(512, latent_textual)
        self.fc_logvar_textual = nn.Linear(512, latent_textual)

        ############

        self.apply(weights_init)
    
    def forward(self, x_textual):
        
        x_textual = self.textual_encoder(x_textual)
        
        mu_textual = self.fc_mu_textual(x_textual)
        logvar_textual = self.fc_logvar_textual(x_textual)

        return mu_textual, logvar_textual
    
#encoder = vae_textual_encoder(latent_textual=128)
#print(summary(encoder, (1, 1024), device="cpu"))


################
# Decoder ######
################


class vae_visual_decoder(nn.Module):
    def __init__(self, latent_visual=128, multimodal = False):
        super(vae_visual_decoder, self).__init__()
        
        self.multimodal = multimodal
        self.latent_visual_in = latent_visual
            
        
        ############
        
        visual_modules = []
        
        # visual decoder
        visual_modules.append(nn.Linear(self.latent_visual_in,512))
        visual_modules.append(nn.ReLU())
        visual_modules.append(nn.Linear(512, 1024))
        visual_modules.append(nn.ReLU())
        visual_modules.append(nn.Linear(1024, 1536))
        visual_modules.append(nn.Sigmoid())
        
        
        self.visual_decoder = nn.Sequential(*visual_modules)
        
        self.apply(weights_init)
                    
    def forward(self, x_visual, x_textual=None):
        
        if self.multimodal:
            x_visual = self.visual_decoder(x_visual)
            x_textual = self.visual_decoder(x_textual)
            
            return x_visual, x_textual

        else:
            x = self.visual_decoder(x_visual)

            return x

#decoder = vae_visual_decoder(latent_visual=128,  multimodal=False) 
#print(summary(decoder, (1,128), device="cpu"))

#decoder = vae_visual_decoder(latent_visual=128,  multimodal=True) 
#print(summary(decoder, [(1,128), (1,128)], device="cpu"))


class vae_textual_decoder(nn.Module):
    def __init__(self, latent_textual=128, multimodal = False):
        super(vae_textual_decoder, self).__init__()
        
        self.multimodal = multimodal
        self.latent_textual_in = latent_textual 
        
        ############
        
        textual_modules = []
        
        # visual encoder
        textual_modules.append(nn.Linear(self.latent_textual_in,128))
        textual_modules.append(nn.ReLU())
        textual_modules.append(nn.Linear(128, 512))
        textual_modules.append(nn.ReLU())
        textual_modules.append(nn.Linear(512, 1024))
        textual_modules.append(nn.Sigmoid())
        
        self.textual_decoder = nn.Sequential(*textual_modules)
        
        self.apply(weights_init)
                    
    def forward(self, x_textual, x_visual=None):
        
        if self.multimodal:
            x_textual = self.textual_decoder(x_textual)
            x_visual = self.textual_decoder(x_visual)
            
            return x_textual, x_visual

        else:
            x = self.textual_decoder(x_textual)

            return x

#decoder = vae_textual_decoder(latent_textual=128,  multimodal=False) 
#print(summary(decoder, (1,128), device="cpu"))

#decoder = vae_textual_decoder(latent_textual=128,  multimodal=True) 
#print(summary(decoder, [(1,128), (1,128)], device="cpu"))

class vae(nn.Module):
    def __init__(self, latent_visual=128, latent_textual=128, multimodal = False, kl_coef = 0.5):
        super(vae, self).__init__()
        
        self.visual_latent_features = latent_visual
        self.textual_latent_features = latent_textual
        self.multimodal = multimodal
        self.kl_coef = kl_coef
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.CE_loss = nn.CrossEntropyLoss()
        
        self.visual_encoder_network = vae_visual_encoder(self.visual_latent_features)
        self.visual_decoder_network = vae_visual_decoder(self.visual_latent_features, self.multimodal)
        
        if self.multimodal:
            self.textual_encoder_network = vae_textual_encoder(self.textual_latent_features)
            self.textual_decoder_network = vae_textual_decoder(self.textual_latent_features, self.multimodal)
        
    def forward(self, x, x_text=None):
        
        network_outputs = dict()
        
        if self.multimodal:
            ###############
            # # visual side
            ###############
            mu_visual, logvar_visual = self.visual_encoder_network(x)
            mu_text, logvar_text = self.textual_encoder_network(x_text)
            
            # latent vector
            z_visual = self.reparameterize(mu_visual, logvar_visual)
            z_text = self.reparameterize(mu_text, logvar_text)
            
            # decoding
            visual_decoded_output_from_visual, visual_decoded_output_from_textual = self.visual_decoder_network(z_visual,z_text)
            textual_decoded_output_from_textual, textual_decoded_output_from_visual = self.textual_decoder_network(z_text,z_visual)
            
            network_outputs["visual_decoded_output_from_visual"] = visual_decoded_output_from_visual
            network_outputs["visual_decoded_output_from_textual"] = visual_decoded_output_from_textual
            network_outputs["textual_decoded_output_from_textual"] = textual_decoded_output_from_textual
            network_outputs["textual_decoded_output_from_visual"] = textual_decoded_output_from_visual
            
            network_outputs["mu_visual"] = mu_visual
            network_outputs["logvar_visual"] = logvar_visual
            network_outputs["mu_text"] = mu_text
            network_outputs["logvar_text"] = logvar_text
        
            return network_outputs
        
        else:
            mu_visual, logvar_visual = self.visual_encoder_network(x)
            
            z_visual = self.reparameterize(mu_visual, logvar_visual)
            
            visual_decoded_output = self.visual_decoder_network(z_visual)
            
            network_outputs["visual_decoded_output"] = visual_decoded_output
            
            network_outputs["mu_visual"] = mu_visual
            network_outputs["logvar_visual"] = logvar_visual
            
            return network_outputs

    # reparameterization trick
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.3).exp_()
        eps = Variable(std.new(std.size()).normal_(0, 1))
        
        return eps.mul(std).add_(mu)
        
    
    # generate sample -from z stage
    def sample(self, size):
        sample = Variable(torch.randn(size, self.latent_features))
        if self.cuda():
            sample = sample.cuda()
        sample = self.decoder_network(sample).cpu()
        return sample
    
    ##############
    # calculate loss functions
    ##############
    def reconstruction_loss(self, x, recon_x):
        # Mean absolute error for reconstruction
        return self.l1_loss(x, recon_x)
    
    
    def KL_loss(self, mu, logvar, kl_beta=0.5):
        # distribution convergence loss
        self.kl = torch.mean(-1 * kl_beta * torch.sum(1 + logvar - mu ** 2 - logvar.exp()), dim = 0)
        
        return self.kl
     
        
    def distribution_alignment_loss(self, mu_1, mu_2, logvar_1, logvar_2):
        distance = torch.sqrt(torch.sum((mu_1 - mu_2) ** 2, dim=1) + torch.sum((torch.sqrt(logvar_1.exp()) - torch.sqrt(logvar_2.exp())) ** 2, dim=1))        
        return distance.sum().mean()


    def dist_aware_classification(self, mus=None, logvars=None, test_mus=None, test_logvars=None, n_way=None, k_shot=None, hidden_size=None, k_shot_test=None, gt_tensor=None):
        
        # find the train elements' mu and logvar values
        # find the test elements' mu and logvar values
        # compare them then assign a label to each test sample
        # according to the gt values, calculate the cross-entropy loss
        # return predicted labels and cross-entropy loss
        
        gt_tensor = gt_tensor.reshape(-1,1)
        #gt_tensor = np.array(gt, dtype=np.int32)
        #gt_tensor = torch.from_numpy(gt_tensor).type(torch.long).reshape(-1,1)

        
        # data preparation for classification
        #####
        mus = mus.reshape((n_way*k_shot, hidden_size))
        logvars = logvars.reshape((n_way*k_shot, hidden_size))
        test_mus = test_mus.reshape((n_way*k_shot_test, hidden_size))
        test_logvars = test_logvars.reshape((n_way*k_shot_test, hidden_size))
        
        # smoothing term
        #####
        alpha = 1 / k_shot
        
        # final classification mu & logvar
        #####
        avg_train_mus = []
        avg_train_vars = []
        
        for i in range(0,len(mus) - 1, k_shot):
            avg_mu = np.mean(mus[i: i + k_shot].cpu().detach().numpy(), axis=0)
            
            class_mu = mus[i: i + k_shot].cpu().detach().numpy()
            class_variance = logvars[i: i + k_shot].cpu().detach().numpy()
                
            # in-class mu and logvar
            avg_var = [0.0] * hidden_size
            for j in range(len(class_variance)):
                          
                if k_shot != 1:
                    avg_var += ((alpha * class_variance[j] ) + (alpha * ((class_mu - avg_mu)**2)))
                # If there is only 1 sample, avg_var equals to that sample's var
                else:
                    avg_var += class_variance[j]        
                    
            # class oriented var
            avg_train_vars.append(avg_var)
            
            # class oriented mu
            avg_train_mus.append(avg_mu)
            
        # testing stage
        #####
        test_pred_labels = []
        ce_loss_total = 0.0
        
        test_mus = test_mus.cpu().detach()
        test_logvars = test_logvars.cpu().detach()

        for i in range(len(test_mus)):
            test_dist = torch.distributions.Normal(test_mus[i],test_logvars[i])

            local_scores = []
            for j in range(len(avg_train_mus)):
                
                # KL divergence
                train_dist = torch.distributions.Normal(torch.Tensor(avg_train_mus[j]), torch.Tensor(avg_train_vars[j]))
                                
                scr = torch.distributions.kl_divergence(test_dist, train_dist).mean()
                local_scores.append(scr)
                            
            local_similarities = torch.from_numpy(np.array([-1 * elem for elem in local_scores])).reshape(1,-1).to(0)
            
            """
            print(local_scores)
            print(local_similarities)
            print(gt_tensor[i])
            print("\n\n")
            
            print(local_similarities.get_device())
            print(gt_tensor[i].get_device())
            """
            # categorical cross entropy loss    
            cc_loss = self.CE_loss(local_similarities, gt_tensor[i])
            
            # min of kl divergence for class decision
            pred_cls_idx = np.argmin(local_scores)
            
            """
            print("----------")
            print(pred_cls_idx)
            print(cc_loss)
            print("----------")
            """
            ce_loss_total += cc_loss
            test_pred_labels.append(pred_cls_idx)
        
        # loss amount and classification accuracy calculation
        # print(classification_report(gt, test_pred_labels))
        
        return {"cross_entropy_loss": ce_loss_total / len(test_mus), "predicted_labels":test_pred_labels}
        
    
# unimodal - visual
#vae_model = vae(latent_visual=128, latent_textual=128, multimodal=False)
#print(summary(vae_model, x=(1,1536), device="cpu"))

# multimodal - visual + textual
#vae_model = vae(latent_visual=128, latent_textual=128, multimodal=True)
#print(summary(vae_model, x=[(1,1536), (1,1024)], device="cpu"))
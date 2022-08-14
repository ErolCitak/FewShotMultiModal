import enum
import os
from matplotlib.pyplot import axis
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
from torch.autograd import Variable
from torchsummary import summary
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import classification_report
import scipy.stats as sps
from zmq import device
from torch.nn.functional import softplus

from torch.distributions.mixture_same_family import MixtureSameFamily
import torch.distributions as D


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
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


class vae_visual_encoder(nn.Module):
    def __init__(self, latent_visual=128):
        super(vae_visual_encoder, self).__init__()
        
        self.visual_latent_in = latent_visual

        
        ############
        
        visual_modules = []
        
        # visual encoder
        visual_modules.append(nn.Linear(1920,960))
        visual_modules.append(nn.BatchNorm1d(960))
        visual_modules.append(nn.ReLU())
        #visual_modules.append(nn.Linear(512, self.visual_latent_in*2))
        #visual_modules.append(nn.ReLU())


        
        self.visual_encoder = nn.Sequential(*visual_modules)
        self.fc_mu_visual = nn.Linear(960, latent_visual)
        self.fc_logvar_visual = nn.Linear(960, latent_visual)
        
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
        textual_modules.append(nn.BatchNorm1d(512))
        textual_modules.append(nn.ReLU(512))
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
        visual_modules.append(nn.Linear(self.latent_visual_in,960))
        visual_modules.append(nn.BatchNorm1d(960))
        visual_modules.append(nn.ReLU())
        visual_modules.append(nn.Linear(960, 1920))
        #visual_modules.append(nn.Sigmoid())
        
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
        textual_modules.append(nn.Linear(self.latent_textual_in,512))
        textual_modules.append(nn.BatchNorm1d(512))
        textual_modules.append(nn.ReLU())
        textual_modules.append(nn.Linear(512, 1024))
        #textual_modules.append(nn.Sigmoid())
        
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
        self.l1_loss = nn.L1Loss(size_average='False')
        self.l2_loss = nn.MSELoss(size_average='False')
        self.CE_loss = nn.CrossEntropyLoss(size_average='False') # reduction="mean"
        
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
            
            network_outputs["z_visual"] = z_visual
            network_outputs["z_text"] = z_text
        
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
        """
        std = logvar.exp_().pow(0.5)
        eps = Variable(std.new(std.size()).normal_(0, 1))
        
        return eps.mul(std).add_(mu)
        """
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        
        return z
            
    ##############
    # calculate loss functions
    ##############
    def reconstruction_loss(self, x, recon_x):
        # Mean absolute error for reconstruction
        return self.l2_loss(x, recon_x)
    
    
    def KL_loss(self, mu, logvar, kl_beta=0.5):
        # distribution convergence loss
        #self.kl = torch.mean(-1 * kl_beta * torch.sum(1 + logvar - (mu.pow(2) ) - logvar.exp()), dim = 0)
        self.kl = -1 * kl_beta * torch.sum(1 + logvar - mu.pow(2)  - logvar.exp())
        
        return self.kl
     
        
    def distribution_alignment_loss(self, mu_1, mu_2, logvar_1, logvar_2, is_batch=True):
        
        if is_batch:
            distance = torch.sqrt(torch.sum((mu_1 - mu_2) ** 2, dim=1) + torch.sum((torch.sqrt(logvar_1.exp()) - torch.sqrt(logvar_2.exp())) ** 2, dim=1))
        else:
            distance = torch.sqrt(torch.sum((mu_1 - mu_2) ** 2, dim=0) + torch.sum((torch.sqrt(logvar_1.exp()) - torch.sqrt(logvar_2.exp())) ** 2, dim=0))
            
        return distance.sum()

    def kl_mvn(self, m0, S0, m1, S1):
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[0]
        iS1 = np.linalg.inv(S1)
        diff = m1 - m0

        # kl is made of three terms
        tr_term   = np.trace(iS1 @ S0)
        det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
        quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
        #print(tr_term,det_term,quad_term)
        return .5 * (tr_term + det_term + quad_term - N) 
    
    def margin_loss_calculator(self, mus, logvars, test_mus, test_logvars, n_way, k_shot, k_shot_test, device="cuda:0"):
        # margin loss
        # ML; aims to take closer of train elements inside together
        margin_loss = torch.zeros((1,)).to(device)
        
        for i in range(0,len(mus), k_shot):
            cls_mus = mus[i: i + k_shot]
            cls_logvars = logvars[i: i + k_shot]
            
            cls_margin = torch.zeros((1,)).to(device)
            n_compare = 0
            for cls_i in range(len(cls_mus)-1):
                for cls_j in range(cls_i+1, len(cls_mus)):
                    cls_margin.add_(self.distribution_alignment_loss(cls_mus[cls_i], cls_mus[cls_j], cls_logvars[cls_i], cls_logvars[cls_j], is_batch=False))
                    n_compare += 1
            
            cls_margin.div_(n_compare)
            margin_loss.add_(cls_margin)
        
        
        
        return margin_loss

    def margin_loss_calculatorv2(self, mus, logvars, avg_train_mus, avg_train_vars, n_way, k_shot, k_shot_test, device="cuda:0"):
            # margin loss
            # ML; aims to take closer of train elements inside together
            margin_loss = torch.zeros((1,)).to(device)
            
            for i in range(0,len(mus), k_shot):
                cls_mus = mus[i: i + k_shot]
                cls_logvars = logvars[i: i + k_shot]
                
                cls_margin = torch.zeros((1,)).to(device)
                n_compare = 0
                for cls_i in range(len(cls_mus)-1):
                    for cls_j in range(cls_i+1, len(cls_mus)):
                        cls_margin.add_(self.distribution_alignment_loss(cls_mus[cls_i], cls_mus[cls_j], cls_logvars[cls_i], cls_logvars[cls_j], is_batch=False))
                        n_compare += 1
                
                cls_margin.div_(n_compare)
                margin_loss.add_(cls_margin)
            
            ##################
            for i in range(0,len(avg_train_mus)-1, 1):                
                for j in range(i+1,len(avg_train_mus), 1):
                    # distance between class i and class j                    
                    ws_distance = self.distribution_alignment_loss(avg_train_mus[i], avg_train_mus[j], avg_train_vars[i], avg_train_vars[j], is_batch=False)
                    
                    margin_loss.add_(ws_distance)
            
            
            return margin_loss
    
    def kl_based_distance_calculator(self, test_mus, test_logvars, avg_train_mus, avg_train_vars, n_way, k_shot_test, device="cuda:0"):
        
        # transfer test_mus & test_logvars to the device
        test_mus = test_mus
        test_logvars = test_logvars
        
        test_pred_labels = torch.mul(torch.ones((len(test_mus)), dtype=torch.uint8), -1)
        local_similarities = []
        
        for i in range(len(test_mus)):
            #t_mu = torch.mean(test_logvars[i])
            #t_var = torch.mean(test_logvars[i])
            #test_dist = torch.distributions.Normal(t_mu, t_var)
            test_dist = torch.distributions.Normal(test_mus[i], test_logvars[i].exp())

            local_scores = []
            for j in range(len(avg_train_mus)):
                
                # KL divergence
                train_dist = torch.distributions.Normal(avg_train_mus[j], avg_train_vars[j])
                
                scr = torch.distributions.kl_divergence(test_dist, train_dist).mean()
                
                local_scores.append(scr)
            
            # min of kl divergence for class decision
            local_scores = torch.stack(local_scores)
            pred_cls_idx = torch.argmin(local_scores)
            test_pred_labels[i] = pred_cls_idx
            
            #local_similarities.append(np.array([-1 * elem for elem in local_scores]).reshape(1,-1))
            local_similarities.append(torch.mul(local_scores, -1))
            
        local_similarities = torch.stack(local_similarities)
        
        
        return [local_similarities, test_pred_labels]
    
    def pdf_sim_calculator(self, test_mus, test_logvars, avg_train_mus, avg_train_vars, n_way, k_shot_test, n_samples=10, device="cuda:0"):
        
        # return variables
        test_pred_labels = []
        test_pdfs = []
        
        # transfer mu & variance to the CPU
        test_mus = test_mus.to(device)
        test_logvars = test_logvars.to(device)
        
        
        for test_i in range(len(test_mus)):
            # generate # n_samples for each test element
            test_dist = torch.distributions.Normal(test_mus[test_i],  test_logvars[test_i])
            samples = test_dist.rsample((n_samples,))
            
            #samples = sps.multivariate_normal.rvs(test_mus[test_i], np.diag(test_logvars[test_i]), size=n_samples)
            
            
            # consists of mean pdf scores; for 3-way --> it has 3 pdfs per each sample
            mean_pdfs = []
             
            # for each sample in a specific test mu&var, find its pdf for each of avg_train_mu&var
            for train_i in range(len(avg_train_mus)):
                
                train_dist = torch.distributions.Normal(avg_train_mus[train_i], avg_train_vars[train_i])
                """
                mean_pdf = []
                for sample in samples:
                    mean_pdf.append(train_dist.log_prob(sample).exp().mean(dim=0))
                mean_pdf = sum(mean_pdf) / len(mean_pdf)
                """
                mean_pdf = train_dist.log_prob(samples).exp().mean(dim=0).mean(dim=-1)
                mean_pdfs.append(mean_pdf)
            
            pred_cls_idx = torch.argmax(torch.Tensor(mean_pdfs)).detach().numpy()
            test_pred_labels.append(pred_cls_idx)
            test_pdfs.append(mean_pdfs)
            
        test_pdfs = torch.tensor(test_pdfs).reshape(n_way*k_shot_test, n_way).to(0)

        return [test_pdfs, test_pred_labels]
    
    def probability_density_calculator_torch_implementation(self, test_zs, train_mus, train_logvars, n_way, k_shot, device="cuda:0"):
        
        # create empty list for likelihoods and predicted labels
        all_samples_pdf_scores = torch.empty((len(test_zs), n_way)).to(device)
        all_samples_predicted_labels = torch.empty(len(test_zs)).to(device)
        
        # create gmm list
        gmm_list = []
        
        for i in range(0,len(train_mus), k_shot):
            cls_mus = train_mus[i: i + k_shot]
            cls_logvars = train_logvars[i: i + k_shot]
            
            mix = D.Categorical(torch.ones(k_shot,).to(device))
            comp = D.Independent(D.Normal(cls_mus, cls_logvars), 1)
            gmm = MixtureSameFamily(mix, comp)
            
            gmm_list.append(gmm)
            

        # for a given each z-latent vector
        # calculate the log_prob of it wrt to the
        # each mixture of gmm
        for test_idx, test_z in enumerate(test_zs):
            sample_pdf_score = [] 
            for cls_idx, mixture_model in enumerate(gmm_list):
                log_prob_score = mixture_model.log_prob(test_z)
                all_samples_pdf_scores[test_idx][cls_idx] = log_prob_score
                sample_pdf_score.append(log_prob_score)
            
            predicted_label = sample_pdf_score.index(max(sample_pdf_score))
            all_samples_predicted_labels[test_idx] = predicted_label
            
        return [all_samples_pdf_scores, all_samples_predicted_labels]

    def mixture_KL(self, test_mus, test_logvars, train_mus, train_logvars, n_way, k_shot, device="cuda:0"):
        
        # create empty list for likelihoods and predicted labels
        all_samples_kl_scores = torch.empty((len(test_mus), n_way)).to(device)
        all_samples_predicted_labels = torch.empty(len(test_mus)).to(device)
        
        # create gmm list
        gmm_list = []
        
        for i in range(0,len(train_mus), k_shot):
            cls_mus = train_mus[i: i + k_shot]
            cls_logvars = train_logvars[i: i + k_shot]
            
            mix = D.Categorical(torch.ones(k_shot,).to(device))
            comp = D.Independent(D.Normal(cls_mus, cls_logvars), 1)
            gmm = MixtureSameFamily(mix, comp)
            
            gmm_list.append(gmm)
            

        # for a given each z-latent vector
        # calculate the log_prob of it wrt to the
        # each mixture of gmm
        for test_idx, test_mu in enumerate(test_mus):
            sample_kl_score = [] 
            for cls_idx, mixture_model in enumerate(gmm_list):
                test_dist = torch.distributions.Normal(test_mus[test_idx], test_logvars[test_idx])
                class_dist = torch.distributions.Normal(gmm_list[cls_idx].mean, gmm_list[cls_idx].variance)
                kl_score = torch.distributions.kl_divergence(test_dist,class_dist).mean()
                all_samples_kl_scores[test_idx][cls_idx] = torch.mul(kl_score, -1)
                sample_kl_score.append(kl_score)
            
            predicted_label = sample_kl_score.index(min(sample_kl_score))
            all_samples_predicted_labels[test_idx] = predicted_label
            
        return [all_samples_kl_scores, all_samples_predicted_labels]
     
    """
        This function takes a latent vector than calculate the pdf of it for all other
            train mu and vars. Then return the raw logits of the likelihood of pdfs 
    """
    def probability_density_calculator(self, test_zs, train_mus, train_logvars, n_way, k_shot, device="cuda:0"):
        
        all_samples_pdf_scores = torch.empty((len(test_zs), n_way)).to(device)
        all_samples_predicted_labels = torch.empty(len(test_zs)).to(device)
        
        
        # for a given each z-latent vector
        for test_idx, test_z in enumerate(test_zs):
            sample_pdf_score = []
            # for each component of a each class
            cls_idx = 0
            for i in range(0,len(train_mus), k_shot):
                cls_mus = train_mus[i: i + k_shot]
                cls_logvars = train_logvars[i: i + k_shot]
                
                cls_pdf = 0.0
                for cls_mu,cls_logvar in zip(cls_mus, cls_logvars):
                    
                    #dist = torch.distributions.Normal(cls_mu, cls_logvar)
                    #pdf_score = dist.log_prob(test_z).exp().mean(dim=0)
                    dist = torch.distributions.MultivariateNormal(cls_mu, torch.diag(cls_logvar))
                    pdf_score = dist.log_prob(test_z)
                    cls_pdf = cls_pdf + ((1/k_shot) * pdf_score)
                
                sample_pdf_score.append(cls_pdf)
                all_samples_pdf_scores[test_idx][cls_idx] = cls_pdf
                cls_idx += 1
            
            predicted_label = sample_pdf_score.index(max(sample_pdf_score))
            all_samples_predicted_labels[test_idx] = predicted_label
            
        return [all_samples_pdf_scores, all_samples_predicted_labels]

    def dist_aware_classification(self, mus=None, logvars=None, z_train=None, z_test=None, test_mus=None, test_logvars=None, n_way=None, k_shot=None, hidden_size=None, k_shot_test=None, gt_tensor=None):
        
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
        
        z_train = z_train.reshape((n_way*k_shot, hidden_size))
        z_test = z_test.reshape((n_way*k_shot_test, hidden_size))        
        
                        
        # final classification mu & logvar
        #####
        
        avg_train_mus = []
        avg_train_vars = []
        avg_train_zs = []
        
        for i in range(0,len(mus), k_shot):
            avg_mu = torch.mean(mus[i: i + k_shot], axis=0)
            avg_var = torch.sum(torch.mul((1/k_shot**2), logvars[i: i + k_shot].exp()), axis=0)
            avg_z = torch.mean(z_train[i: i + k_shot], axis=0)
                   
            # class oriented mu & var 
            avg_train_mus.append(avg_mu)
            avg_train_vars.append(avg_var)
            avg_train_zs.append(avg_z)
        
        avg_train_mus = torch.stack(avg_train_mus)
        avg_train_vars = torch.stack(avg_train_vars)
        avg_train_zs = torch.stack(avg_train_zs)
        
        
        # calculate the margin loss
        margin_loss = self.margin_loss_calculator(mus, logvars, test_mus, test_logvars, n_way, k_shot, k_shot_test)
        #margin_loss = self.margin_loss_calculatorv2(mus, logvars, avg_train_mus, avg_train_vars, n_way, k_shot, k_shot_test)
        #margin_loss = 0.0
        
        # testing stage
        #####
        # 1) KL-divergence based    
        kl_div_result = self.kl_based_distance_calculator(test_mus, test_logvars, avg_train_mus, avg_train_vars, n_way, k_shot_test)
        local_similarities, test_pred_labels = kl_div_result[0], kl_div_result[1]
        
        # 2) PDF similarity based
        #pdf_sim_result = self.pdf_sim_calculator(test_mus, test_logvars, avg_train_mus, avg_train_vars, n_way, k_shot_test, n_samples=10)
        #local_similarities, test_pred_labels = pdf_sim_result[0], pdf_sim_result[1]
        
        # 3) Probability Density Based - Torch
        #pdf_sim_result = self.probability_density_calculator_torch_implementation(z_test, mus, logvars, n_way, k_shot)
        #local_similarities, test_pred_labels = pdf_sim_result[0], pdf_sim_result[1]
        
        # 4) Mixture - KL
        #pdf_sim_result = self.mixture_KL(test_mus, test_logvars, mus, logvars, n_way, k_shot)
        #local_similarities, test_pred_labels = pdf_sim_result[0], pdf_sim_result[1]
        
        # ground truth values
        gt_tensor = gt_tensor.reshape(n_way*k_shot_test)
        
        # categorical cross entropy loss    
        cc_loss = self.CE_loss(local_similarities, gt_tensor)
        
        # kl_loss directly
        """
        cc_loss = torch.zeros((1,),dtype=torch.float64).to("cuda:0")
        # add minimum constant
        j = 0
        for i in range(0,len(test_mus), k_shot_test):
            
            class_err = local_similarities[i:i+k_shot_test][:,j]*-1
            class_err = class_err.mean()
            
            cc_loss += class_err
            
            j += 1
        
        # get the average cc_los; for example, 3-way 5-shot, 15 shots per class in test
        # cc_loss /= (3*15)
        cc_loss = torch.div(cc_loss, n_way * k_shot_test)
        """

            
        return {"cross_entropy_loss": cc_loss, "margin_loss":margin_loss, "predicted_labels":test_pred_labels}
        
    
# unimodal - visual
#vae_model = vae(latent_visual=128, latent_textual=128, multimodal=False)
#print(summary(vae_model, x=(1,1536), device="cpu"))

# multimodal - visual + textual
#vae_model = vae(latent_visual=128, latent_textual=128, multimodal=True)
#print(summary(vae_model, x=[(1,1536), (1,1024)], device="cpu"))
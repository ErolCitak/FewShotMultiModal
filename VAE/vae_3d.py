import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torchvision
from torch.autograd import Variable

from torchsummary import summary

import os
import numpy as np

def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class vae_encoder(nn.Module):
    def __init__(self, latent_features=128):
        super(vae_encoder, self).__init__()
        
        # definition of the 3D-CNN Blocks
        self.conv_layer1 = self._conv_layer_set(3, 32) # 56
        self.conv_layer2 = self._conv_layer_set(32, 32) # 28
        self.conv_layer3 = self._conv_layer_set(32, 64) # 14
        self.conv_layer4 = self._conv_layer_set(64, 64) # 7
        
        self.fc_mu = nn.Linear(1*7*7*64, latent_features)
        self.fc_logvar = nn.Linear(1*7*7*64, latent_features)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        
        return conv_layer
    
    def forward(self, x):
        
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        
        x = x.view(-1, 1*7*7*64)
        mu = F.relu(self.fc_mu(x))
        logvar = F.relu(self.fc_logvar(x))
        
        return mu, logvar

#encoder = vae_encoder(latent_features=128)
#print(summary(encoder,(3,16,112,112), device="cpu"))



################
# Decoder ######
################


class vae_decoder(nn.Module):
    def __init__(self, latent_features=128):
        super(vae_decoder, self).__init__()
        
        self.fc_latent = nn.Linear(latent_features, 64*1*7*7)
        
        # definition of the 3D-CNN Blocks
        self.deconv_layer1 = self._deconv_layer_set(64, 1) # 14
        self.deconv_layer2 = self._deconv_layer_set(1, 64) # 28
        self.deconv_layer3 = self._deconv_layer_set(64, 32) # 56
        self.deconv_layer4 = self._deconv_layer_set(32, 3) # 112
        
        self.sigmoid = nn.Sigmoid()
        

    def _deconv_layer_set(self, in_c, out_c):
        deconv_layer = nn.Sequential(
        nn.ConvTranspose3d(in_c, out_c, kernel_size=(2,2,2), stride=(2,2,2)),
        nn.LeakyReLU()
        )
        
        return deconv_layer
    
    def forward(self, x):
        
        
        x = self.fc_latent(x)
        x = x.view(-1,64,1, 7, 7)
        
        x = self.deconv_layer1(x)
        x = self.deconv_layer2(x)
        x = self.deconv_layer3(x)
        x = self.deconv_layer4(x)
        
        x = self.sigmoid(x)
    
        
        return x

#decoder = vae_decoder(latent_features=128)
#print(summary(decoder, (1,128), device="cpu"))

class vae_text_encoder(nn.Module):
    def __init__(self, text_latent_features=128):
        super(vae_text_encoder, self).__init__()
        
        self.text_latent_features = text_latent_features
        
        self.fc1 = nn.Linear(512, 256)
        self.mu = nn.Linear(256, self.text_latent_features)
        self.logvar = nn.Linear(256, self.text_latent_features)
            
    
    def forward(self, x):
        
        # encode text info.
        x = F.relu(self.fc1(x))
        
        text_mu = F.relu(self.mu(x))
        text_logvar = F.relu(self.logvar(x))
        
        return text_mu, text_logvar 
    
#encoder = vae_text_encoder(text_latent_features=128)
#print(summary(encoder, (1,512), device="cpu"))


class vae_text_decoder(nn.Module):
    def __init__(self, text_latent_features=128):
        super(vae_text_decoder, self).__init__()
        
        self.text_latent_features = text_latent_features
        
        self.fc1 = nn.Linear(self.text_latent_features, 256)
        self.decoded_text_output = nn.Linear(256, 512)
            
    def forward(self, x):
        
        # decode text info.
        x = F.relu(self.fc1(x))
        
        decoded_text = self.decoded_text_output(x)
        
        return decoded_text 
        
#decoder = vae_text_decoder(text_latent_features=128)
#print(summary(decoder, (1,128), device="cpu"))


class vae(nn.Module):
    def __init__(self, latent_features=128, multimodal = False, kl_coef = 0.5):
        super(vae, self).__init__()
        
        self.latent_features = latent_features
        self.multimodal = multimodal
        self.kl_coef = kl_coef
        
        self.encoder_network = vae_encoder(latent_features=128)
        self.decoder_network = vae_decoder(latent_features=128)
        
        if self.multimodal:
            self.text_encoder_network = vae_text_encoder(text_latent_features=128)
            self.text_decoder_network = vae_text_decoder(text_latent_features=128)
        
    def forward(self, x, x_text=None):
        ###############
        # # visual side
        ###############
        mu, logvar = self.encoder_network(x)
        
        # latent vector
        z = self.reparameterize(mu, logvar)
        
        # decoding
        decoded_output = self.decoder_network(z)
        
        ##############
        # # text side
        ##############
        if self.multimodal:
            mu_text, logvar_text = self.text_encoder_network(x_text)
            
            # latent vector
            z_text = self.reparameterize(mu_text, logvar_text)
            
            # decoding
            decoded_text_output = self.text_decoder_network(z_text)
            
            
            return decoded_output, decoded_text_output
        
        else:
            return decoded_output

    # reparameterization trick
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.new(std.size()).normal_())
        
        return eps.mul(std).add_(mu)

    # generate sample -from z stage
    def sample(self, size):
        sample = Variable(torch.randn(size, self.latent_features))
        if self.cuda():
            sample = sample.cuda()
        sample = self.decoder_network(sample).cpu()
        return sample

    # calculate loss function
    def loss_function(self, x, recon_x, mu, logvar):
        # Mean absolute error for reconstruction
        self.mae = F.l1_loss(recon_x, x, reduction='none')
        self.mae = torch.div(self.mae,torch.numel(x))
        self.mae = torch.sum(self.mae)

        # distribution convergence loss
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return self.mae + self.kl_coef*self.kl

    
# unimodal - visual
#vae_model = vae(latent_features=128, multimodal=False)
#print(summary(vae_model, x=(3,16,112,112), device="cpu"))

# multimodal - visual + textual
#vae_model = vae(latent_features=128, multimodal=True)
#print(summary(vae_model, x=(3,16,112,112), x_text=(1,512), device="cpu"))
from threading import Timer
from tkinter import E
import numpy as np
import os
import shutil
import ast
import random
from timeit import timeit
from torch.nn.functional import one_hot

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Dataset Loader
class HVUDataset():
    def __init__(self, folder_path, n_way, k_shot, n_test_per_class=1, multimodal=True, transform=None):
        self.folder_path = folder_path
        self.classes = os.listdir(self.folder_path)
        self.nb_classes = len(self.classes) 
        self.n_way = n_way
        self.k_shot = k_shot
        self.multimodal = multimodal
        self.transform = transform
        self.n_test_per_class = n_test_per_class
        self.normalizer = MinMaxScaler()
        
        self.samples_per_class = [os.listdir(os.path.join(self.folder_path, action_class)) for action_class in self.classes]
   
    # this function will return one n-way, k-shot set
    def next_batch(self):
        
        # select class idx randomly; for example 1, 18, 24, 56 for 4-way
        selected_classes_idx = random.sample(range(0, self.nb_classes), self.n_way)
        
        # select items for each selected class above
        selected_data_visual = []
        selected_data_visual_test = []
        
        y_selected_data_visual = []
        y_selected_data_visual_test = []
        
        
        if self.multimodal:
            selected_data_textual = []
            selected_data_textual_test = []
            
            y_selected_data_textual = []
            y_selected_data_textual_test = []
            
        # select k+n_test_per_class samples per class; --> n_batch . (n-way. (k-shot + n_test_per_class))
        for order_inter_class, selected_class_idx in enumerate(selected_classes_idx):
            
            visual_features = [None] * (self.k_shot + self.n_test_per_class)
            y_visual_features = [None] * (self.k_shot + self.n_test_per_class)
            
            if self.multimodal:
                textual_features = [None] * (self.k_shot + self.n_test_per_class)
                y_textual_features = [None] * (self.k_shot + self.n_test_per_class)
            
            class_path = os.path.join(self.folder_path, self.classes[selected_class_idx])
            n_per_class = len(os.listdir(class_path))
            
            class_samples_idx = random.sample(range(0, n_per_class), self.k_shot + self.n_test_per_class)
            
            for order_intra_class, idx in enumerate(class_samples_idx):
                sample_path = os.path.join(class_path, os.listdir(class_path)[idx])

                # read visual and textual feature (if needed)
                visual_feature = np.load(os.path.join(sample_path, "DenseNet_201.npz"))['DenseNet_201']
                visual_feature = self.normalizer.fit_transform(visual_feature.reshape(-1, 1)).reshape(1,-1)
                
                visual_feature = torch.FloatTensor(visual_feature)
                visual_features[order_intra_class] = visual_feature
                y_visual_features[order_intra_class] = torch.IntTensor(np.ones(1) * order_inter_class)
                
                if self.multimodal:
                    textual_feature = np.load(os.path.join(sample_path, "Elmo_Mean.npz"))['Elmo']
                    textual_feature = self.normalizer.fit_transform(textual_feature.reshape(-1, 1)).reshape(1,-1)
                    
                    textual_feature = torch.FloatTensor(textual_feature)
                    textual_features[order_intra_class] = textual_feature
                    y_textual_features[order_intra_class] = torch.IntTensor(np.ones(1) * order_inter_class)
            
            # output per batch is organized as follows;
            """
                n_way, k_shot: 3-way 5-shot
                selected_data_visual: [ [class_1_samples], [class_2_samples], [class_3_samples] ]
                if multimodal: selected_data_textual: [ [class_1_samples], [class_2_samples], [class_3_samples]]
            """
            #selected_data_visual[order_inter_class] = visual_features
            selected_data_visual.extend(visual_features[:self.k_shot])
            selected_data_visual_test.extend(visual_features[self.k_shot:])
            
            y_selected_data_visual.extend(y_visual_features[:self.k_shot])
            y_selected_data_visual_test.extend(y_visual_features[self.k_shot:])
            
            if self.multimodal:
                #selected_data_textual[order_inter_class] = textual_features
                selected_data_textual.extend(textual_features[:self.k_shot])
                selected_data_textual_test.extend(textual_features[self.k_shot:])
                    
                y_selected_data_textual.extend(y_textual_features[:self.k_shot])
                y_selected_data_textual_test.extend(y_textual_features[self.k_shot:])


            #txt_feature = ast.literal_eval((self.annotations_text.iloc[index, 2]))
            #txt_feature = torch.FloatTensor(txt_feature)
        
        
        if self.multimodal:
            return {'visual_data_train': selected_data_visual, 'visual_data_train_label': y_selected_data_visual,
                    'visual_data_test': selected_data_visual_test, 'visual_data_test_label': y_selected_data_visual_test,
                    'textual_data_train': selected_data_textual, 'textual_data_train_label': y_selected_data_textual,
                    'textual_data_test': selected_data_textual_test, 'textual_data_test_label': y_selected_data_textual_test,
                    #'selected_classes': selected_classes_idx
                    }
        else:
            return {'visual_data_train': selected_data_visual, 'visual_data_train_label': y_selected_data_visual,
                    'visual_data_test': selected_data_visual_test, 'visual_data_test_label': y_selected_data_visual_test}
        
   
# Test it
if __name__ == "__main__":
    n_way = 3
    k_shot = 5
    n_test_per_class = 1
    
    dataset = HVUDataset("C:/HolisticVideoUnderstanding/uniform_train", n_way= n_way, k_shot= k_shot, n_test_per_class=n_test_per_class, multimodal=True, transform=None)

    # time consumption
    #print(timeit(dataset.next_batch, number=100))
    
    print("{}-way -- {}-shot\n{}-test samples per class".format(n_way, k_shot, n_test_per_class))
    
    class_frequency = dict()
    
    
    
    dataloader_result = dataset.next_batch()

    visual_x, visual_y, visual_test_x, visual_test_y, textual_x, textual_y, textual_test_x, textual_test_y \
        = dataloader_result["visual_data_train"], dataloader_result["visual_data_train_label"] \
        , dataloader_result["visual_data_test"], dataloader_result["visual_data_test_label"] \
        , dataloader_result["textual_data_train"], dataloader_result["textual_data_train_label"] \
        , dataloader_result["textual_data_test"], dataloader_result["textual_data_test_label"]
    
    for elem in visual_x:
        b = elem
        print(torch.mean(b),torch.min(b),torch.max(b))
    
    print("----------------")
    for elem in textual_x:
        b = elem
        print(torch.mean(b),torch.min(b),torch.max(b))

    """
    print("----->Visual")
    print("n-Visual Train_x: {}".format(len(visual_x)))
    print("n-Visual Train_y: {}".format(len(visual_y)))
    print("n-Visual Test_x: {}".format(len(visual_test_x)))
    print("n-Visual Test_y: {}".format(len(visual_test_y)))
    print("----->Textual")
    print("n-Textual Train_x: {}".format(len(textual_x)))
    print("n-Textual Train_y: {}".format(len(textual_y)))
    print("n-Textual Test_x: {}".format(len(textual_test_x)))
    print("n-Textual Test_y: {}".format(len(textual_test_y)))
    print("-------------------")
    print(visual_x)
    b = torch.stack(visual_x)
    print(b.shape)
    print(len(b))
    print(b)
    print("-------------------")
    print(visual_y)
    c = torch.stack(visual_y)
    d = torch.stack(visual_test_y)
    c_encoded = one_hot(torch.tensor(visual_y).long())
    d_encoded = one_hot(torch.tensor(visual_test_y).long())
    print(c)
    print(d) 
    print(c_encoded)
    print(d_encoded)
    print(c_encoded.shape)
    """
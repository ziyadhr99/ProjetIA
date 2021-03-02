# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 15:34:38 2021rain

@author: Big-BOSS
"""
DIM_L = 28
DIM_C = 28
BLOC_L = 4 
BLOC_C = 4

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch as th
from torchvision import datasets,transforms
import math

# Récupérer les données d'apprentissage
TrainSet = datasets.MNIST(root='./resources', train=True, download=True)
TestSet = datasets.MNIST(root='./resources', train=False, download=True)

# Determiner les dimensions
print(TrainSet)
print(TestSet)

# Resize an image 
from skimage.transform import resize


# spliting TrainSet into 4 subgroups for taining and validation
from sklearn.model_selection import train_test_split
train_data, validation_data, train_targets, validation_targets = train_test_split(TrainSet.data, TrainSet.targets, test_size=0.33)

# shaping the test data 
test_data = TestSet.data
test_targets = TestSet.targets

#Normalisation des données 

# Convertir les valeurs des tensors en float
train_data = train_data.float()
test_data = test_data.float()
validation_data = validation_data.float()


def parametre(dim_x,dim_y,bloc_x,bloc_y):
            i = int(dim_x/bloc_x)
            j = int(dim_y/bloc_y)
            lx = bloc_x*[i]
            ly = bloc_y*[j]
            x = bloc_x*[0]
            y = bloc_y*[0]
            if dim_x%bloc_x > 0 :
                for k in range(dim_x%bloc_x):
                    x[k] = 1
            if dim_y%bloc_y > 0 :
                for k in range(dim_y%bloc_y):
                    y[k] = 1
            for k in range(bloc_x):
                lx[k] = lx[k] + x[k]
            for k in range(bloc_y): 
                ly[k] = ly[k] + y[k]
            for k in range(1,bloc_x):
                lx[k] += lx[k-1]
            for k in range(1,bloc_y):
                ly[k] += ly[k-1]
            return [0]+lx,[0]+ly
        


DIM_BLOC_L , DIM_BLOC_C = parametre(DIM_L, DIM_C , BLOC_L , BLOC_C)


# Importer Dataset de torch.utils.data
from torch.utils.data import Dataset
from skimage.morphology import skeletonize
# Créer une classe qui hérite de Dataset et redéfinit les méthodes comme susmentionné

class MyDataset(Dataset):

    def empty_line(self,T):
            for i in T:
                if i>0: 
                    return False
            return True
    def delete_empty_line(self,photo):       #supprimer les lignes nulles
            length = len(photo)
            l = []
            init = 0
            out = length
            step = 1
            for i in range(2):
                for i in range(init , out ,step ):
                    if self.empty_line(photo[i]) :         #empty line
                        l += [i]
                    else : 
                        break
                    if((length - len(l))<=6):
                        break   
                init = -1
                out = -length
                step = -1
            photo = np.delete(photo,l,axis=0)
            return photo
    def delete_empty_column(self,photo):
            photo = np.transpose(photo)
            photo = self.delete_empty_line(photo)
            return np.transpose(photo)
    
    def image_processing(self,photo):
            photo = self.delete_empty_line(photo)
            photo = self.delete_empty_column(photo)
            photo = resize(photo,(28,28))
            photo = (photo - photo.mean())/photo.std()
            m = (np.max(photo)+np.min(photo))/2
            photo = np.where(photo>m,1,0)
            photo = skeletonize(photo)
            photo = np.where(photo==True,1,0)
            return photo
        
    def block(self,photo):                   #diviser l'image en blocks
            l = [ [ [] for j in range(BLOC_C) ] for i in range(BLOC_L) ]   # j colonne i ligne
            for i in range(BLOC_L):
                for j in range(BLOC_C):
                        l[i][j] = photo[DIM_BLOC_L[i]:DIM_BLOC_L[i+1],DIM_BLOC_C[j]:DIM_BLOC_C[j+1]]
            return l
        
    def pix_pos(self,bloc):                # nombre de pixel positif dans un bloc
            s = 0 
            for i in range(len(bloc)):
                for j in range(len(bloc[0])):
                    if bloc[i][j] != 0 :
                        s+=1
            return s
        
    def first_param(self,photo):           #le premier paramètre de featuring
            blocks = self.block(photo)
            pix_p = []
            for i in range(BLOC_L):
                for j in range(BLOC_C):
                    pix_p += [self.pix_pos(blocks[i][j])]
            return pix_p/ np.sum(pix_p)
    def coeff_Reg(self,bloc):                    #retourne le coefficient b d'un bloc
            X = len(bloc)
            Y = len(bloc[0])        
            ab = 0
            XX = []
            YY = []
            for i in range(-1,-X-1,-1):
                for j in range(Y):
                    if bloc[i][j]==1:
                        YY += [ab]
                        XX += [j]
                ab += 1
    
            if math.isnan(np.cov(XX,YY)[0][1]/(np.std(XX)**2)): return 0        
            else : return np.cov(XX,YY)[0][1]/(np.std(XX)**2)
    
    def coeff_bloc(self,photo):                      #returne une liste des coefficients b pour chaque bloc
            blocks = self.block(photo)
            coeff_b = []
            for i in range(BLOC_L):
                for j in range(BLOC_C):
                    coeff_b += [self.coeff_Reg(blocks[i][j])]
            return coeff_b   
    
    def second_third_param(self,photo):              #les 2eme et 3eme parametres de featuring
            coeff_b = self.coeff_bloc(photo)
            second = []
            third = []
            for i in range(len(coeff_b)):
                second += [(2*coeff_b[i])/(1+coeff_b[i]**2)]
                third += [(1- coeff_b[i]**2)/(1+coeff_b[i]**2)]
            return second , third
    
    def feature_param(self,photo):
            photo = self.image_processing(photo)
            first = self.first_param(photo)
            second, third = self.second_third_param(photo)
            feature = []
            for i in range(len(first)):
                feature += [first[i],second[i],third[i]]
            return feature
        
    def extraction(self,data):
        input = [ [] for i in range(len(data)) ]
        for i in range(len(data)):
            print(i)
            input[i] = self.feature_param(data[i].numpy())
        return th.tensor(input)
    
    def __init__(self, data=None , targets=None):
        super(MyDataset, self)
        if data != None :
            self.data = self.extraction(data)
            self.targets = targets
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    def __len__(self):
        return len(self.targets)



train_dataset = MyDataset(train_data, train_targets)
test_dataset = MyDataset(test_data, test_targets)
validation_dataset = MyDataset(validation_data, validation_targets)


# Importer DataLoader de torch.utils.data
from torch.utils.data import DataLoader
# Créer une variable pour la taille du batch
batch_size = 64
# Créer les objets DataLoader pour vos datasets d'apprentissage, test et validation en lui donner la taille du batch convenue
train_DL = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_DL = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
validation_DL = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


# Importer le module nn
import torch.nn as nn
# Importer torch.nn.functional
import torch.nn.functional as F

# Créer une classe qui hérite de nn.Module et redéfinir le constructeur ainsi que la méthode forward
class Net (nn.Module):
    def __init__ (self):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(BLOC_C*BLOC_L*3,20)
        self.hidden_layer = nn.Linear(20,15)
        self.output_layer = nn.Linear(15,10)
    def forward(self, x):        
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x 
    
# Instancier votre NN
net = Net()


# Définir la fonction du coût. On peut choisir CrossEntropyLoss (It is useful when training a classification problem with C classes)
loss_function = nn.CrossEntropyLoss()
# Définir une fonction d'optimisation des coût: Adam par exemple. On devra définir un learning rate. On choisira 0.001.
import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Définir le nombre d'epochs. Commencer petit. 
epoch = 15

# boucle d'apprentissage:
for ep in range(epoch):
    
    net.train()
    train_loss = 0
    
    for feature, target in train_DL:
        
        output = net(feature.float())
        loss = loss_function(output, target)
        
        #initialiser l'optimizer
        optimizer.zero_grad()
        #backpropagation
        loss.backward()
        #effectuer un pas
        optimizer.step()
        
        train_loss +=  loss.item() #it's a tensor
    
    train_loss /= len(train_DL)
        

    net.eval()
    # boucle de validation:
    valid_loss = 0
    correct = 0
    total = 0
    
    with th.no_grad():

        for feature, target in validation_DL:
                        
            output = net(feature.float())
            
            loss = loss_function(output, target)
            valid_loss +=  loss.item()
            correct += th.sum(th.argmax(output, dim=1) == target).item()

        valid_loss /= len(validation_DL)
        correct /= len(validation_DL.dataset) #10.000

    print(f"epoch: {ep}, train loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}, correct predictions: {correct*100:.2f}%")  


# boucle de test:
test_loss = 0
correct = 0

with th.no_grad():
    
    for data in test_DL:
        feature, target = data
        
        output = net(feature.float())
        
        loss = loss_function(output, target)
        test_loss += loss.item()
        
        correct += th.sum(th.argmax(output, dim=1) == target).item()
        

    test_loss /= len(test_DL)
    correct /= len(test_DL.dataset) #10.000  

print(f"Accuracy {correct*100:.2f}%")







import PIL
from PIL import ImageTk, Image, ImageDraw
from tkinter import *
from tkinter.ttk import Scale
from tkinter import filedialog , messagebox
import PIL.ImageGrab as ImageGrab
import win32gui
import matplotlib.pyplot as plt

class Paint():
    def __init__(self,root):
        self.data = MyDataset()
        self.root = root
        self.root.title("Projet IA")
        self.root.geometry("400x250")
        self.root.configure(background='white')
        self.root.resizable(0,0)
        
        self.save_button = Button(self.root,text="Predict",bd=2,bg="white",command=self.save_photo,width=8,relief='ridge')
        self.save_button.place(x=2,y=20)
        
        self.clear_button = Button(self.root,text="Clear",bd=2,bg="white",command=self.clear ,width=8,relief='ridge')
        self.clear_button.place(x=2,y=60)
        
        #Create pen
        self.pen_size_scale_frame = LabelFrame(self.root,text="size",bd=5,bg='white',font=('arial',15,'bold'),relief='ridge')
        
        self.pen_size = Scale()
        self.pen_size.set(1)
        
        #•Create canvas (espace de painting)
        self.canvas = Canvas(self.root,bg='white',bd=5,relief=GROOVE,height=220,width=310)


        self.canvas.place(x=75,y=15)
        
        self.canvas.bind('<B1-Motion>',self.paint)
        
        self.image1 = PIL.Image.new("RGB", (310,220), (0,0,0))
        self.draw = ImageDraw.Draw(self.image1)
        
    def paint(self,event):
        x1,y1 = (event.x -4),(event.y - 4)
        x2,y2 = (event.x +4),(event.y + 4)
        
        self.canvas.create_oval(x1,y1,x2,y2,fill="black",outline="black",width=self.pen_size.get())
        self.draw.line([x1, y1, x2, y2],fill="white",width=5)
    def clear(self):
        self.canvas.delete("all")
        self.image1 = PIL.Image.new("RGB", (310,220), (0,0,0))
        self.draw = ImageDraw.Draw(self.image1)
    def save_photo(self):
        try:
            filename = "image.jpg"
            self.image1.save(filename)

            imgpaint = np.asarray(self.image1)
            imgpaint = imgpaint[:,:,2]
            input1 = th.tensor(self.data.feature_param(imgpaint)).float()
            print(th.argmax(net(input1)))
            messagebox.showinfo('Projet IA','la valeur prédite est '+ str(th.argmax(net(input1))) )
        except:
            print("something went wrong, plz try again")
            

if __name__ == '__main__':
    root = Tk()
    p = Paint(root)
    root.update()
    root.mainloop()





import os

path = os.getcwd()
parent = os.path.dirname(path)
for dir in os.listdir(parent+'/data/Animal-Seg-V3/train'):
    
    if dir!='.DS_Store' and len(os.listdir(parent+'/data/Animal-Seg-V3/train/'+dir))==1:
        for file in os.listdir(parent+'/data/Animal-Seg-V3/val/'+dir):
            os.rename(parent+'/data/Animal-Seg-V3/val/'+dir+'/'+file,parent+'/data/Animal-Seg-V3/train/'+dir+'/'+file)
        
        
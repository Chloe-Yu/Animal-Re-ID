import os

path = os.getcwd()
parent = os.path.dirname(path)
folders = ['Animal-Seg-V3','Animal-2']
for folder in folders:
    for dir in os.listdir(parent+'/data/'+folder+'/train'):
        if dir!='.DS_Store' and len(os.listdir(parent+'/data/'+folder+'/train/'+dir))==1:
            for file in os.listdir(parent+'/data/'+folder+'/val/'+dir):
                os.rename(parent+'/data/'+folder+'/val/'+dir+'/'+file,parent+'/data/'+folder+'/train/'+dir+'/'+file)
            
            
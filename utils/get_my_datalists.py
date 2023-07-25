import os

with open('train-yak-ori.txt','r') as f, open('yak_mytrain.txt','w') as train, open('yal_myval.txt','w') as val:
    for line in f.readlines():
        path,ent,pos = line.split(' ')
        _,fol,name = path.split('/')
        fol = str(int(fol)+300) #for yak
        imgs = list(os.listdir('/home/yinyu/Thesis/data/Animal-Seg-V3/val/'+fol+'/'))
        imgs_train = list(os.listdir('/home/yinyu/Thesis/data/Animal-Seg-V3/train/'+fol+'/'))
        assert len(imgs)==1
        img = imgs[0]
        if name != img:
            #train.write(f"train/{fol}/{name} {ent} {pos}")
            if name not in imgs_train:
                print(line)
            else:
                train.write(f"train/{fol}/{name} {fol} {pos}")
        else:
            #val.write(f"val/{fol}/{name} {ent} {pos}")
            val.write(f"val/{fol}/{name} {fol} {pos}")
        
    
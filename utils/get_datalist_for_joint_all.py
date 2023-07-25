with open('/home/yinyu/Thesis/animal-reid/datalist/all_train_aligned.txt','r') as f, open('/home/yinyu/Thesis/animal-reid/datalist/yak_train_all.txt','w') as train:
    for line in f.readlines():
        path,ent,pos = line.split(' ')
        if int(ent)>227 or int(ent)<107:
            ent = '-'+ent
        train.write(f"{path} {ent} {pos}")
        
    
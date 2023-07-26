with open('/home/yinyu/Thesis/animal-reid/datalist/ele_train_all.txt','w') as f, open('/home/yinyu/Thesis/animal-reid/datalist/mytrain.txt','r') as tiger, \
    open('/home/yinyu/Thesis/animal-reid/datalist/yak_mytrain_aligned.txt','r') as yak, open('/home/yinyu/Thesis/animal-reid/datalist/ele_train.txt','r') as ele:
    for line in tiger.readlines():
        path,ent,pos = line.split(' ')
        f.write(f"{path} {int(ent)+337} {pos}")
    
    for line in yak.readlines():
        path,ent,pos = line.split(' ')
        f.write(f"{path} {int(ent)+444} {pos}")
        
    f.write(f"\n")
    for line in ele.readlines():
        path,ent,pos = line.split(' ')
        f.write(f"{path} {int(ent)} {pos}")
        
    
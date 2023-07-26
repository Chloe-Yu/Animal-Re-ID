with open('/home/yinyu/Thesis/animal-reid/datalist/all_val_aligned_m.txt','w') as f, open('/home/yinyu/Thesis/animal-reid/datalist/myval.txt','r') as tiger, \
    open('/home/yinyu/Thesis/animal-reid/datalist/yak_myval_aligned.txt','r') as yak, open('/home/yinyu/Thesis/animal-reid/datalist/ele_val.txt','r') as ele:
    for line in tiger.readlines():
        path,ent,pos = line.split(' ')
        f.write(f"{path} {ent} {pos}")
    
    for line in yak.readlines():
        path,ent,pos = line.split(' ')
        f.write(f"{path} {int(ent)+107} {pos}")
        
    for line in ele.readlines():
        path,ent,pos = line.split(' ')
        f.write(f"{path} {int(ent)+228} {pos}")
        
    
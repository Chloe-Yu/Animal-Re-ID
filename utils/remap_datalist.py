import os

with open('all_val.txt','r') as f, open('all_val_aligned.txt','w') as out:
    txt = f.readlines()
    ents = []
    for line in txt:
        path, ent, pos = line.split(' ')
        ents.append(int(ent))
    
    ents = list(set((ents)))
    class_to_inx = {d:i for i,d in enumerate(sorted(ents))}
    #print(class_to_inx)
    
    for line in txt:
        path, ent, pos = line.split(' ')

        out.write(f"{path} {class_to_inx[int(ent)]} {pos}")

        
    
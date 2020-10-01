from tqdm import tqdm
from helpers import centerUpdate
import torch
from torch.nn import CrossEntropyLoss


def trainer(model, loader, optimizer, loss_fn, cfg, centers, ep):
    
    train_loss, center_distances, c_classes = train(model, loader['train'], optimizer, loss_fn, cfg, centers, ep)
    val_loss = eval(model, loader['val'], optimizer, loss_fn, cfg, centers, center_distances, ep)
    
    if cfg['loss'] != 'ce':
        centerUpdate(centers, c_classes, cfg['class_num'])

    return train_loss, val_loss
    
    
def train(model, loader, optimizer, loss_fn, cfg, centers, ep):

    model.train()

    center_distances = dict()
    c_classes = dict()
    
    if cfg['loss'] != 'ce':
        for classId in range(cfg['class_num']):      
            diffClassInd = [x for x in range(cfg['class_num']) if x != classId]
            distances = ((centers[classId]-centers[diffClassInd])**2).sum(dim=-1)
            center_distances[classId] = centers[diffClassInd][distances.argmin()]

    with tqdm(total=len(loader), dynamic_ncols=True) as progress:
        progress.set_description(f'Epoch: {ep+1}')
        t_loss = 0

        for idx, (inputs, target) in enumerate(loader):

            optimizer.zero_grad()
            
            if cfg['previousModel']:

                with torch.no_grad():
                    ins = cfg['previousModel'].forward(inputs.cuda())
                    
                out = model.forward(ins.cuda()).reshape(ins.size()[0], -1)
            
            else:
                out = model.forward(inputs.cuda()).reshape(inputs.size()[0], -1)
            
            if cfg['loss'] == 'ce':
                loss = loss_fn(out, target.cuda())
            else:
                loss = loss_fn(out, target, cfg['m'], centers, cfg['class_num'], c_classes, center_distances)

            loss.backward()
            optimizer.step()

            t_loss += loss.item()

            avg_loss = t_loss / (idx + 1)
            progress.update(1)
            progress.set_postfix(loss=avg_loss)

    return t_loss / len(loader), center_distances, c_classes


def eval(model, loader, optimizer, loss_fn, cfg, centers, center_distances, ep):

    model.eval()
    
    with torch.no_grad():
        with tqdm(total=len(loader), dynamic_ncols=True) as progress:
        
            progress.set_description(f'Eval: {ep+1}')
            v_loss = 0

            for idx, (inputs, target) in enumerate(loader):

                if cfg['previousModel']:
                    ins = cfg['previousModel'].forward(inputs.cuda())
                    out = model.forward(ins.cuda()).reshape(ins.size()[0], -1)
                
                else:
                    out = model.forward(inputs.cuda()).reshape(inputs.size()[0], -1)
                
                if cfg['loss'] == 'ce':
                    loss = loss_fn(out, target.cuda())
                else:
                    loss = loss_fn(out, target, cfg['m'], centers, cfg['class_num'], None, center_distances)

                v_loss += loss.item()

                avg_loss_v = v_loss / (idx + 1)
                progress.update(1)
                progress.set_postfix(loss=avg_loss_v)
        
    return v_loss / len(loader)

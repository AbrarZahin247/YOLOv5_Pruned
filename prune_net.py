from utils.torch_utils import prune,select_device,de_parallel
from utils.general import intersect_dicts
from pathlib import Path
from models.yolo import Model
import torch
import datetime
# import yaml


def prune_model(weights_path,save_dir='pruned_models',pruning_ratio=0.5,anchors=3,device=0,batch_size=8,nc=7,resume=False,anchor_path="data/hyps/hyp.scratch-low.yaml"):
    ckpt = torch.load(weights_path, map_location="cpu")    
    device=select_device(device, batch_size=batch_size)
    model = Model(ckpt["model"].yaml, ch=3, nc=nc, anchors=anchors).to(device)  # create
    exclude = ["anchor"] if (anchors) and not resume else []  # exclude keys
    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    
    w = Path(save_dir) / "weights"
    w.mkdir(parents=True, exist_ok=True)
    sparced = w / "pruned.pt"
    
    prune(model=model,amount=pruning_ratio)
    from copy import deepcopy
    ckpt = {
                    "model": deepcopy(de_parallel(model)).half()
            }
    # Save last, best and delete
    torch.save(ckpt, sparced)

weight_path="runs/train/yolov5n_okutama_slimming26/weights/best.pt"
pruning_ratio=0.6
device=1
batch_size=16
prune_model(weights_path=weight_path,pruning_ratio=pruning_ratio,device=device,batch_size=batch_size)
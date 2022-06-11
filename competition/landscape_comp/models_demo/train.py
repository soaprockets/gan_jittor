

import torch
import losses as losses
from model_demo import *
import utils as utils
from utils.utils import *
# from utils.fid_scores import fid_pytorch
import config
from dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
#--- read options ---#
opt = config.read_arguments(train=True)
# from models import *
#--- create utils ---#
timer = timer(opt)
visualizer_losses = losses_saver(opt)
losses_computer = losses.losses_computer(opt)
dataroot="/root/autodl-tmp/Data/land_scanp"
transform =transforms.Compose([
    transforms.Resize(size=(384, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset_train =ImageDataset(root=dataroot,transforms=transform)
train_loader=DataLoader(dataset_train,batch_size=4,shuffle=True,drop_last=True)
im_saver = image_saver(opt)
# fid_computer = fid_pytorch(opt, None)

#--- create models ---#

model = OASIS_model(opt)
model = put_on_multi_gpus(model, opt)

#--- create optimizers ---#
optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

#--- the training loop ---#
already_started = False
device=torch.device('cuda:0')
start_epoch, start_iter = get_start_iters(opt.loaded_latest_iter, len(train_loader))
for epoch in range(start_epoch, opt.num_epochs):
    for i, data_i in enumerate(train_loader):
        if not already_started and i < start_iter:
            continue
        already_started = True
        cur_iter = epoch*len(train_loader) + i
        image, label = preprocess_input(opt, data_i)
        image,label = image.to(device),label.to(device)

        #--- generator update ---#
        model.module.netG.zero_grad()
        loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()

        #--- discriminator update ---#
        model.module.netD.zero_grad()
        loss_D, losses_D_list = model(image, label, "losses_D", losses_computer)
        loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
        loss_D.backward()
        optimizerD.step()

        #--- stats update ---#
        if not opt.no_EMA:
            update_EMA(model, cur_iter, train_loader, opt)
        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model, image, label, cur_iter)
            timer(epoch, cur_iter)
        if cur_iter % opt.freq_save_ckpt == 0:
            save_networks(opt, cur_iter, model)
        if cur_iter % opt.freq_save_latest == 0:
            save_networks(opt, cur_iter, model, latest=True)
        # if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
        #     # is_best = fid_computer.update(model, cur_iter)
        #     if is_best:
        #         utils.save_networks(opt, cur_iter, model, best=True)
        visualizer_losses(cur_iter, losses_G_list+losses_D_list)

#--- after training ---#
update_EMA(model, cur_iter, train_loader, opt, force_run_stats=True)
save_networks(opt, cur_iter, model)
save_networks(opt, cur_iter, model, latest=True)
# is_best = fid_computer.update(model, cur_iter)
# if is_best:
#     utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")


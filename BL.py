import math
import os
import time
import numpy as np
import torch
import clip
from itertools import cycle
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.utils.data
from metric import psnr, ssim
from data import RESIDE_Dataset, TestDataset, CLIP_loader
from CLIP import L_clip_from_feature
from collections import OrderedDict
from option.BL import opt
from res2netmodel import syn_model
from architect.architect_uv import Architect

start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.loss_fn = nn.SmoothL1Loss()
        self.encoder = syn_model.Res2Net(syn_model.Bottle2neck, [3, 4, 23, 3])
        self.decoder = syn_model.decoder(res_blocks=18)

    def forward(self, x):
        x_e = self.encoder(x)
        out = self.decoder(x, x_e)
        return out

    def loss_syn(self, x, y):
        out_syn = self.forward(x)
        l1_loss = self.loss_fn(out_syn, y)
        total_syn_loss = opt.w_loss_L1 * l1_loss
        return l1_loss, total_syn_loss

    def loss_rea(self, z, r):
        out_rea = self.forward(z)
        clip_loss = Clip_loss(out_rea, text_features)
        total_rea_loss = opt.w_loss_CLIP * clip_loss
        return clip_loss, total_rea_loss


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


def lr_schedule_cosdecay_encoder(t, T, init_lr=opt.start_sgd_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def lr_schedule_cosdecay_decoder(t, T, init_lr=opt.start_adam_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def train(Model, loader_train_1, loader_train_2, loader_test, Model_encoder_optim, Model_decoder_optim):
    losses = []
    loss_log = {'L1': []}
    loss_log_tmp = {'L1': []}
    psnr_log = []

    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    loader_train_iter_1 = iter(loader_train_1)
    loader_train_iter_2 = cycle(iter(loader_train_2))

    for step in range(start_step + 1, steps + 1):
        Model.train()

        encoder_lr = opt.start_sgd_lr
        decoder_lr = opt.start_adam_lr

        if not opt.no_lr_sche:

            encoder_lr = lr_schedule_cosdecay_encoder(step, T)
            for param_group in Model_encoder_optim.param_groups:
                param_group["lr"] = encoder_lr

            decoder_lr = lr_schedule_cosdecay_decoder(step, T)
            for param_group in Model_decoder_optim.param_groups:
                param_group["lr"] = decoder_lr

        try:
            x, y = next(loader_train_iter_1)
        except StopIteration:
            loader_train_iter_1 = iter(loader_train_1)
            x, y = next(loader_train_iter_1)
        try:
            z, r = next(loader_train_iter_2)
        except StopIteration:
            loader_train_iter_2 = iter(loader_train_2)
            z, r = next(loader_train_iter_2)

        x = x.to(opt.device)
        y = y.to(opt.device)
        z = z.to(opt.device)
        r = r.to(opt.device)

        Model_decoder_optim.zero_grad()
        architect.unrolled_backward(x, y, z, r, encoder_lr, Model_encoder_optim)
        Model_decoder_optim.step()

        Model_encoder_optim.zero_grad()
        l1_loss, loss = Model.loss_syn(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(Model.encoder.parameters(), 5.0)
        Model_encoder_optim.step()

        losses.append(loss.item())

        loss_log_tmp['L1'].append(opt.w_loss_L1 * l1_loss.item())

        print(
            f'\r| L1:{opt.w_loss_L1 * l1_loss.item():.5f} | step :{step}/{steps} | encoder_lr :{encoder_lr :.9f} | decoder_lr :{decoder_lr :.9f} | time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        if step % len(loader_train_1) == 0:
            loader_train_iter_1 = iter(loader_train_1)
            for key in loss_log.keys():
                loss_log[key].append(np.average(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = []
            np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)
        if (step % opt.iters_per_epoch == 0 and step <= opt.finer_eval_step) or (
                step > opt.finer_eval_step and (step - opt.finer_eval_step) % (5 * len(loader_train_1)) == 0):
            if step > opt.finer_eval_step:
                epoch = opt.finer_eval_step // opt.iters_per_epoch + (step - opt.finer_eval_step) // (
                        5 * len(loader_train_1))
            else:
                epoch = int(step / opt.iters_per_epoch)
            with torch.no_grad():
                ssim_eval, psnr_eval = test(Model, loader_test)

            log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f} | encoder_lr:{encoder_lr:.12f} | decoder_lr:{decoder_lr:.12f}'
            print(log)
            with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)
            state_dict = Model.state_dict()
            if 'module' in list(state_dict.keys())[0]:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                state_dict = new_state_dict
            if psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                print(
                    f'model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')
                saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pth')
                torch.save(state_dict, saved_best_model_path)
            saved_single_model_path = os.path.join(opt.saved_model_dir, str(epoch) + '.pth')
            torch.save(state_dict, saved_single_model_path)
            loader_train_iter_1 = iter(loader_train_1)
            np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
            np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    for i, (inputs, targets, hazy_name) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            H, W = inputs.shape[2:]
            inputs = pad_img(inputs, 4)
            pred = net(inputs).clamp(0, 1)
            pred = pred[:, :, :H, :W]
        ssim_tmp = ssim(pred, targets).item()
        psnr_tmp = psnr(pred, targets)
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)

    return np.mean(ssims), np.mean(psnrs)


def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    set_seed_torch(2024)
    clip_model, _ = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")  # ViT-B/32
    clip_model.to(opt.device)
    for param in clip_model.parameters():
        param.requires_grad = False
    res_model, _ = clip.load("RN101", device=torch.device("cpu"), download_root="./clip_model/")
    res_model.to(opt.device)
    for param in res_model.parameters():
        param.requires_grad = False
    data = torch.load('./clip_model/haze_prompt.pth')
    new_state_dict = OrderedDict()
    for k, v in data.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    embedding_prompt = nn.Parameter(new_state_dict['embedding_prompt']).cuda()
    text_encoder = TextEncoder(clip_model)
    embedding_prompt.requires_grad = False
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * 16)]])
    text_features = text_encoder(embedding_prompt, tokenized_prompts)
    clip_model.eval()
    res_model.eval()

    Model = Model().to(opt.device)

    Clip_loss = L_clip_from_feature().to(opt.device)

    Model.load_state_dict(torch.load('./weight/Pre.pth', map_location=torch.device("cpu")))

    train_dir_1 = '../dataset/THaze/train'
    train_set_1 = RESIDE_Dataset(train_dir_1, True, 256, '.jpg')

    test_dir = '../dataset/THaze/val'
    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'), '.jpg')

    train_dir_2 = '../dataset/Real'
    train_set_2 = CLIP_loader(train_dir_2, True, 256)

    loader_train_1 = DataLoader(dataset=train_set_1, batch_size=24, shuffle=True, num_workers=8)
    loader_train_2 = DataLoader(dataset=train_set_2, batch_size=8, shuffle=True, num_workers=2)
    loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

    Model_encoder_optim = optim.SGD(params=filter(lambda x: x.requires_grad, Model.encoder.parameters()), lr=opt.start_sgd_lr,
                                    momentum=opt.momentum, weight_decay=opt.weight_decay)
    Model_decoder_optim = optim.Adam(params=filter(lambda x: x.requires_grad, Model.decoder.parameters()), lr=opt.start_adam_lr,
                                     betas=opt.betas, weight_decay=opt.weight_decay, eps=opt.eps)

    Model_encoder_optim.zero_grad()

    architect = Architect(Model, opt)

    train(Model, loader_train_1, loader_train_2, loader_test, Model_encoder_optim, Model_decoder_optim)

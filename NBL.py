import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.utils.data
from metric import psnr, ssim
from data import RESIDE_Dataset, TestDataset
from option.NBL import opt
from res2netmodel import syn_model

start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps


class Pre_Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Pre_Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # out_channel=512
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Syn_model(nn.Module):
    def __init__(self):
        super(Syn_model, self).__init__()
        self.loss_fn = nn.L1Loss()
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


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def train(Syn, loader_train_1, loader_test, Syn_optim):
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

    for step in range(start_step + 1, steps + 1):
        Syn.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in Syn_optim.param_groups:
                param_group["lr"] = lr

        x, y = next(loader_train_iter_1)
        x = x.to(opt.device)
        y = y.to(opt.device)

        if step == 10000:
            print("Step exceeds 10000, unfreezing encoder parameters...")

            for param in Syn.encoder.parameters():
                param.requires_grad = True

            Syn_optim = optim.Adam(params=filter(lambda p: p.requires_grad, Syn.parameters()), lr=lr,
                                   betas=opt.betas, eps=opt.eps)

        Syn_optim.zero_grad()
        l1_loss, loss = Syn.loss_syn(x, y)
        loss.backward()

        Syn_optim.step()

        losses.append(loss.item())

        loss_log_tmp['L1'].append(l1_loss.item())

        print(
            f'\r| L1:{opt.w_loss_L1 * l1_loss.item():.5f} | step :{step}/{steps} | lr :{lr :.9f} | time_used :{(time.time() - start_time) / 60 :.1f}',
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
                ssim_eval, psnr_eval = test(Syn, loader_test)

            log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f} | lr:{lr:.12f}'
            print(log)
            with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)
            state_dict = Syn.state_dict()
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

    Syn = Syn_model().to(opt.device)
    res2net101 = Pre_Res2Net(syn_model.Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
    res2net101.load_state_dict(torch.load('./pre_model/res2net101_v1b_26w_4s-0812c246.pth'))
    pretrained_dict = res2net101.state_dict()
    model_dict = Syn.encoder.state_dict()
    key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(key_dict)
    Syn.encoder.load_state_dict(model_dict)

    for param in Syn.encoder.parameters():
        param.requires_grad = False

    train_dir_1 = '../dataset/THaze/train'
    train_set_1 = RESIDE_Dataset(train_dir_1, True, 256, format='.jpg')

    test_dir = '../dataset/THaze/val'
    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'), format='.jpg')

    loader_train_1 = DataLoader(dataset=train_set_1, batch_size=24, shuffle=True, num_workers=8)
    loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

    Syn_optim = optim.Adam(params=filter(lambda x: x.requires_grad, Syn.decoder.parameters()), lr=opt.start_lr,
                           betas=opt.betas, eps=opt.eps)

    Syn_optim.zero_grad()
    train(Syn, loader_train_1, loader_test, Syn_optim)

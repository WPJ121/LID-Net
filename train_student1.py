import os
import argparse
import json
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pytorch_msssim import ssim, MS_SSIM
from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
import logging
import sys
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.models import vgg16
from perceptual import LossNetwork
from thop import profile
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-light', type=str, help='model name')
parser.add_argument('--model_teacher', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='OTS', type=str, help='dataset name')
parser.add_argument('--exp', default='outdoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPUs used for training')
parser.add_argument('--resume', type=bool, default=True, help='Continue Train')
parser.add_argument('--model_teacher_checkpoint_dir', default='./saved_models/outdoor/dehazeformer-s.pth', type=str, help='checkpoint of TFD')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
global logger


class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        self.log = open(fileN, 'a')

    def write(self, message):
        '''print实际相当于sys.stdout.write'''
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()
    losses_rec = AverageMeter()
    losses_KD1 = AverageMeter()
    losses_KD2 = AverageMeter()
    losses_KD3 = AverageMeter()
    losses_KD4 = AverageMeter()
    losses_KD5 = AverageMeter()
    losses_KDout = AverageMeter()
    losses_msssim= AverageMeter()
    losses_pec= AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in tqdm(train_loader) :
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(args.no_autocast):
            output_stu,stu_layer1_feature, stu_layer2_feature, stu_layer3_feature, stu_layer4_feature, stu_layer5_feature = network(source_img)
            output_t,t_layer1_feature, t_layer2_feature, t_layer3_feature, t_layer4_feature, t_layer5_feature=network_teacher(source_img)

            loss_rec = criterion[0](output_stu, target_img)
            loss_KD1=  criterion[0](stu_layer1_feature,t_layer1_feature)
            loss_KD2 = criterion[0](stu_layer2_feature,t_layer2_feature)
            loss_KD3 = criterion[0](stu_layer3_feature,t_layer3_feature)
            loss_KD4 = criterion[0](stu_layer4_feature,t_layer4_feature)
            loss_KD5 = criterion[0](stu_layer5_feature,t_layer5_feature)
            loss_KDout= criterion[0](output_stu,output_t)

            loss_ssim =1 - criterion[3](output_stu, target_img)
            loss_pec = criterion[4](output_stu, target_img)   #VGG16的感知损失
            # if epoch<5:
            #     loss = loss_rec+loss_rec+loss_pec+loss_ssim+loss_KD4
            #     #print(epoch,'now the trained KD layer is :loss_KD4 ')
            # else:
            #     loss=loss_rec+loss_pec+loss_ssim+loss_KD3+loss_KDout
                #print(epoch,'now the trained KD layer is :loss_KD3, loss_KDout')
            loss = loss_rec + loss_pec + loss_ssim + loss_KD3 + loss_KDout

             #+  loss_KD4 +loss_ssim    #+ loss_KD5 loss_KD +  loss_KD3 +

        losses.update(loss.item())
        losses_rec.update(loss_rec.item())
        losses_KD1.update(loss_KD1.item())
        losses_KD2.update(loss_KD2.item())
        losses_KD3.update(loss_KD3.item())
        losses_KD4.update(loss_KD4.item())
        losses_KD5.update(loss_KD5.item())
        losses_KDout.update(loss_KDout.item())
        losses_msssim.update(loss_ssim.item())
        losses_pec.update(loss_pec.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # 调用 GradScaler 的 backward() 方法计算梯度并缩放
        scaler.step(optimizer)
        scaler.update()

    return (losses.avg, losses_rec.avg, losses_KD1.avg, losses_KD2.avg, losses_KD3.avg,
            losses_KD4.avg, losses_KD5.avg, losses_KDout.avg, losses_msssim.avg, losses_pec.avg)


def valid(val_loader, network,lpips_metric):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPS = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()


        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img)[0].clamp_(-1, 1)

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

        _, _, H, W = output.size()
        down_ratio = max(1, round(min(H, W) / 256))
        ssim_value = ssim(F.adaptive_avg_pool2d(output * 0.5 + 0.5, (int(H / down_ratio), int(W / down_ratio))),
                          F.adaptive_avg_pool2d(target_img * 0.5 + 0.5, (int(H / down_ratio), int(W / down_ratio))),
                          data_range=1, size_average=False).mean()
        SSIM.update(ssim_value.item(), source_img.size(0))

        with torch.no_grad():
            lpips = lpips_metric(output, target_img)
            LPIPS.update(lpips, source_img.size(0))

    return PSNR.avg, SSIM.avg, LPIPS.avg


def model_structure(model):
    blank = ' '
    print('-' * 100)
    print('|' + ' ' * 21 + 'weight name' + ' ' * 20 + '|' \
          + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 100)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 50:
            key = key + (50 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 30:
            shape = shape + (30 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 100)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 100)


def model_mac(model):
    model_path =( os.path.join(save_dir, args.model+ '.pth'))
    model =eval(args.model.replace('-', '_'))() # 模型结构
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.to('cuda')

    # 计算模型的总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # 打印每一层的参数量
    # for name, param in model.named_parameters():
    #    print(f"{name}: {param.numel()} parameters")

    # 使用torchprofile计算模型的计算量
    input_data = torch.randn(2, 3, 256, 256)  # 生成输入数据
    input_data = input_data.to('cuda')  # 将输入数据移动到GPU上
    # macs= profile_macs(model,input_data)  # 输入尺寸根据实际情况调整
    macs, params = profile(model, inputs=(input_data,))
    print('MACs计算量: % .4fG' % (macs / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值



if __name__ == '__main__':
    global logger
    logger = logging.getLogger()  # pylint: disable=invalid-name
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    t = time.strftime("-%Y%m%d-%H%M%S", time.localtime())  # 时间戳
    sys.stdout = Logger('student '+args.model + t + '.txt')#把print输出为txt文件

    network_teacher=eval(args.model_teacher.replace('-', '_'))()
    network_teacher=nn.DataParallel(network_teacher).cuda()

    network = eval(args.model.replace('-', '_'))()  # eval作用：接收运行一个字符串表达式，返回表达式的结果值。
    # dehazeformer_s()#此处没作用，只是为了体现上一行，方便引进这个函数
    network = nn.DataParallel(network).cuda()

    logger.info(model_structure(network))  #输出网络结构

    start_time = time.time()

    vgg_model = vgg16(pretrained=True)
    vgg_model = vgg_model.features[:16].cuda()
    for param in vgg_model.parameters():
        param.requires_grad = False

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

    criterion = []
    criterion.append(nn.L1Loss().cuda())
    criterion.append(nn.MSELoss().cuda())   #L2损失函数
    criterion.append(nn.SmoothL1Loss().cuda())
    criterion.append(MS_SSIM().cuda())
    criterion.append(loss_network)


    scaler = GradScaler()  # 是一个用于自动混合精度训练的 PyTorch 工具，它可以帮助加速模型训练并减少显存使用量。

    dataset_dir = os.path.join(args.data_dir, args.dataset)  # /data+OTS
    train_dataset = PairLoader(dataset_dir, 'train', 'train',  # /data+/OTS+/Train   路径不区分大小写
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)  # /saved_models+ /outdoor
    os.makedirs(save_dir, exist_ok=True)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()  ##LPIPS 图像相似性度量标准

    ckp_network_teacher = torch.load(args.model_teacher_checkpoint_dir)
    try:
        network_teacher.load_state_dict(ckp_network_teacher['state_dict'])  #torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中

        print(f'The teacher model that has successfully loaded the parameters into the student model is : {args.model_teacher}')
    except:
        print('No weight loaded')

    for name, param in network_teacher.named_parameters():    #named_parameters()将会打印每一次迭代元素的名字和param。
        param.requires_grad = False

    #losses = []
    start_epoch = 0
    best_psnr = 0
    best_ssim = 0
    best_epoch=0
    total_psnr_avg = 0
    total_ssim_avg = 0
    best_lpips = 1
    best = 0
    psnrs = []
    ssims = []
    lpips = []
    total_lr=[]
    use_lr=0
    total_loss=[]

    if args.resume and os.path.exists(os.path.join(save_dir, args.model + '.pth')):  ##resume重新加载参数   saved_models\saved_models\exp   +\TFD
        ckp = torch.load(os.path.join(save_dir, args.model + '.pth'))  ##加载模型
        network.load_state_dict(ckp['state_dict'])  ##加载模型参数。即预训练权重
        start_epoch = ckp['best_epoch']
        best_epoch = ckp['best_epoch']
        best_psnr = ckp['best_psnr']
        best_ssim = ckp['best_ssim']
        ssims = ckp['ssims']
        psnrs = ckp['psnrs']
        total_ssim_avg = ckp['total_ssim_avg']
        total_psnr_avg = ckp['total_psnr_avg']
        total_lr=ckp['total_lr']
        use_lr=ckp['use_lr']
        lpips = ckp['lpips']

        print(f'start_step: {start_epoch} continue to train ---')

    else:
        use_lr=setting['lr']
        print('==> Start training from scratch,, current model name: ' + args.model)
    # print(network)

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=use_lr)
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=use_lr)
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)  # 余弦退火调整学习率

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))  # logs\outdoor\dehazeformer-s
    # 将条目直接写入 log_dir 中的事件文件以供 TensorBoard 使用

    for epoch in tqdm(range(start_epoch + 1, setting['epochs'] + 1)):

        loss ,loss_rec,loss_KD1,loss_KD2,loss_KD3,loss_KD4,loss_KD5,loss_KDout,loss_msssim,loss_pec= train(train_loader, network, criterion, optimizer, scaler)

        writer.add_scalars('train_loss', {'loss':loss,'loss_rec':loss_rec,'loss_KD1':loss_KD1,
                                          'loss_KD2':loss_KD2,'loss_KD3':loss_KD3,'loss_KD4':loss_KD4,'loss_KD5':loss_KD5,
                                          'loss_KDout':loss_KDout,'loss_msssim':loss_msssim,'loss_pec':loss_pec}, epoch)  # 增加一个数据图，坐标对应为loss和epoch
        total_lr.append(optimizer.param_groups[0]["lr"])
        use_lr=optimizer.param_groups[0]["lr"]
        total_loss.append(loss)

        print(
            f'\rEpoch of training now: {epoch} |  | Train loss: {loss:.5f} |  | lr: {optimizer.param_groups[0]["lr"]:.6f} | time_used: {(time.time() - start_time) / 60 :.1f}'
            f'\nloss_KD1:{loss_KD1:5f} || loss_KD2:{loss_KD2:5f} || loss_KD3:{loss_KD3:5f} || loss_KD4:{loss_KD4:5f} || loss_KD5:{loss_KD5:5f}'
            f'\nloss_rec:{loss_rec:5f} || loss_KDout:{loss_KDout:5f} || loss_msssim:{loss_msssim:5f}|| loss_pec:{loss_pec:5f}',
            end='', flush=True)

        scheduler.step()  ##scheduler是对优化器的学习率进行调整

        if epoch % setting['eval_freq'] == 0:
            avg_psnr, avg_ssim,avg_lpips = valid(val_loader, network, lpips_metric)
            total_psnr_avg = total_psnr_avg + avg_psnr
            total_ssim_avg = total_ssim_avg + avg_ssim

            print(f'\nEpoch: {epoch} | psnr: {avg_psnr:.4f} | ssim: {avg_ssim:.4f} | lpips: {avg_lpips:.4f} ')
            writer.add_scalars('valid',{'valid_psnr':avg_psnr,'valid_ssim':avg_ssim,'valid_lpips':avg_lpips,}, epoch)

            psnrs.append(avg_psnr)
            ssims.append(avg_ssim)
            lpips.append(avg_lpips)

            if (avg_psnr > best_psnr):
                best_psnr = avg_psnr
                best_ssim = avg_ssim
                best_lpips = avg_lpips
                best_epoch=epoch

                torch.save({'state_dict': network.state_dict(),  # network.state_dict(),为保存的模型参数
                            'best_epoch': best_epoch,
                            'best_psnr': best_psnr,
                            'best_ssim': best_ssim,
                            'best_lpips':best_lpips,
                            'psnrs': psnrs,
                            'ssims': ssims,
                            'lpips': lpips,
                            'total_ssim_avg': total_ssim_avg,
                            'total_psnr_avg': total_psnr_avg,
                            'use_lr':use_lr,
                            'total_lr':total_lr,
                            'total_loss':total_loss,
                            },#psnrs-total_loss之间的参数，都是保存到训练得到这个最好模型之前的参数，并不是保存所有epoch的参数
                           os.path.join(save_dir, args.model + '.pth'))
                print(
                    f'\n Models saved at best_epoch: {best_epoch} | best_psnr: {best_psnr:.4f} | best_ssim: {best_ssim:.4f}  | best_lpips: {best_lpips:.4f}')
            writer.add_scalar('best_psnr', best_psnr, epoch)

    total_psnr_avg = total_psnr_avg / setting['epochs']
    total_ssim_avg = total_ssim_avg / setting['epochs']

    #model_mac(network)
    logger.info(model_mac(network))



    print(
        f'\nFinished Training Model:{args.model} | best_epoch: {best_epoch} | best_psnr: {best_psnr:.4f} | best_ssim: {best_ssim:.4f} | best_lpips: {best_lpips:.4f} | total_psnr_avg: {total_psnr_avg:.4f} | total_ssim_avg: {total_ssim_avg:.4f} ')



import os
import argparse
from model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import shutil
import models.analysis as analysis
import models.synthesis as synthesis
import models.analysis_prior as analysis_prior
import models.synthesis_prior as synthesis_prior
import models.synthesis_slimmable as synthesis_slimmable
import pruner.SlimmablePruner as SlimmablePruner
from datasets import Datasets, TestKodakDataset, TestCLICPDataset
from Meter import AverageMeter
from tqdm import tqdm
from model_slimmable import ImageCompressor_slimmable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
cal_step = 40
warmup_step = 0#  // gpu_num
parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')

parser.add_argument('-p', '--pretrain', default = '', help='load pretrain model')
parser.add_argument('--config', dest='config', required=False, help = 'hyperparameter in json format')


def parse_config(args):
    config = json.load(open(args.config))
    return config

def log_string(str):
    logger.info(str)
    print(str)

def testKodak(args, config):
    test_dataset = TestKodakDataset(data_dir='../datasets/kodak')
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4)

    Encoder1 = ImageCompressor_slimmable(out_channel_N=config['out_channel_N'],out_channel_M=config['out_channel_M'])
    Encoder2 = ImageCompressor_slimmable(out_channel_N=config['out_channel_N'],out_channel_M=config['out_channel_M'])
    Encoder3 = ImageCompressor_slimmable(out_channel_N=config['out_channel_N'],out_channel_M=config['out_channel_M'])
    Encoder4 = ImageCompressor_slimmable(out_channel_N=config['out_channel_N'],out_channel_M=config['out_channel_M'])
    print("loading models from {}".format(args.pretrain))
    checkpoint = torch.load(args.pretrain)
    Encoder1_state_dict = checkpoint['Encoder1_state_dict']
    Encoder1.load_state_dict(Encoder1_state_dict)
    Encoder2_state_dict = checkpoint['Encoder2_state_dict']
    Encoder2.load_state_dict(Encoder2_state_dict)
    Encoder3_state_dict = checkpoint['Encoder3_state_dict']
    Encoder3.load_state_dict(Encoder3_state_dict)
    Encoder4_state_dict = checkpoint['Encoder4_state_dict']
    Encoder4.load_state_dict(Encoder4_state_dict)

    Decoder1 = synthesis_slimmable.Synthesis_net_slimmable(out_channel_N=int(config['out_channel_N']*config['decoder_width1']),
                                            out_channel_M=int(config['out_channel_M']*1))
    Decoder2 = synthesis_slimmable.Synthesis_net_slimmable(out_channel_N=int(config['out_channel_N']*config['decoder_width2']),
                                            out_channel_M=int(config['out_channel_M']*1))
    Decoder3 = synthesis_slimmable.Synthesis_net_slimmable(out_channel_N=int(config['out_channel_N']*config['decoder_width3']),
                                            out_channel_M=int(config['out_channel_M']*1))
    checkpoint = torch.load(args.pretrain)
    Decoder1_state_dict = checkpoint['Decoder1_state_dict']
    Decoder1.load_state_dict(Decoder1_state_dict)
    Decoder2_state_dict = checkpoint['Decoder2_state_dict']
    Decoder2.load_state_dict(Decoder2_state_dict)
    Decoder3_state_dict = checkpoint['Decoder3_state_dict']
    Decoder3.load_state_dict(Decoder3_state_dict)

    gate21 = SlimmablePruner.Gate()
    gate31 = SlimmablePruner.Gate()
    gate32 = SlimmablePruner.Gate()
    checkpoint = torch.load(args.pretrain)
    gate21_state_dict = checkpoint['gate21_state_dict']
    gate21.load_state_dict(gate21_state_dict)
    gate31_state_dict = checkpoint['gate31_state_dict']
    gate31.load_state_dict(gate31_state_dict)
    gate32_state_dict = checkpoint['gate32_state_dict']
    gate32.load_state_dict(gate32_state_dict)

    Encoder_AQL1 = SlimmablePruner.AQL(192)
    Encoder_AQL2 = SlimmablePruner.AQL(192)
    Encoder_AQL3 = SlimmablePruner.AQL(192)
    checkpoint = torch.load(args.pretrain)
    Encoder_AQL1_state_dict = checkpoint['Encoder_AQL1_state_dict']
    Encoder_AQL1.load_state_dict(Encoder_AQL1_state_dict)
    Encoder_AQL2_state_dict = checkpoint['Encoder_AQL2_state_dict']
    Encoder_AQL2.load_state_dict(Encoder_AQL2_state_dict)
    Encoder_AQL3_state_dict = checkpoint['Encoder_AQL3_state_dict']
    Encoder_AQL3.load_state_dict(Encoder_AQL3_state_dict)

    Decoder_IAQL1 = SlimmablePruner.IAQL(192)
    Decoder_IAQL2 = SlimmablePruner.IAQL(192)
    Decoder_IAQL3 = SlimmablePruner.IAQL(192)
    checkpoint = torch.load(args.pretrain)
    Decoder_IAQL1_state_dict = checkpoint['Decoder_IAQL1_state_dict']
    Decoder_IAQL1.load_state_dict(Decoder_IAQL1_state_dict)
    Decoder_IAQL2_state_dict = checkpoint['Decoder_IAQL2_state_dict']
    Decoder_IAQL2.load_state_dict(Decoder_IAQL2_state_dict)
    Decoder_IAQL3_state_dict = checkpoint['Decoder_IAQL3_state_dict']
    Decoder_IAQL3.load_state_dict(Decoder_IAQL3_state_dict)

    priorEncoder_IAQL1 = SlimmablePruner.IAQL(192)
    priorEncoder_IAQL2 = SlimmablePruner.IAQL(192)
    priorEncoder_IAQL3 = SlimmablePruner.IAQL(192)
    checkpoint = torch.load(args.pretrain)
    priorEncoder_IAQL1_state_dict = checkpoint['priorEncoder_IAQL1_state_dict']
    priorEncoder_IAQL1.load_state_dict(priorEncoder_IAQL1_state_dict)
    priorEncoder_IAQL2_state_dict = checkpoint['priorEncoder_IAQL2_state_dict']
    priorEncoder_IAQL2.load_state_dict(priorEncoder_IAQL2_state_dict)
    priorEncoder_IAQL3_state_dict = checkpoint['priorEncoder_IAQL3_state_dict']
    priorEncoder_IAQL3.load_state_dict(priorEncoder_IAQL3_state_dict)

    priorDecoder_AQL1 = SlimmablePruner.AQL(192)
    priorDecoder_AQL2 = SlimmablePruner.AQL(192)
    priorDecoder_AQL3 = SlimmablePruner.AQL(192)
    checkpoint = torch.load(args.pretrain)
    priorDecoder_AQL1_state_dict = checkpoint['priorDecoder_AQL1_state_dict']
    priorDecoder_AQL1.load_state_dict(priorDecoder_AQL1_state_dict)
    priorDecoder_AQL2_state_dict = checkpoint['priorDecoder_AQL2_state_dict']
    priorDecoder_AQL2.load_state_dict(priorDecoder_AQL2_state_dict)
    priorDecoder_AQL3_state_dict = checkpoint['priorDecoder_AQL3_state_dict']
    priorDecoder_AQL3.load_state_dict(priorDecoder_AQL3_state_dict)

    '''TEST FULL MODEL'''
    print('-------------------------------------')
    print('TEST FULL MODEL')
    print('-------------------------------------')
    # TEST FIRST BPP
    Encoder4 = Encoder4.cuda()
    Encoder_AQL1.cuda()
    Decoder_IAQL1.cuda()
    priorEncoder_IAQL1.cuda()
    priorDecoder_AQL1.cuda()
    max_psnr_diff = 0
    # TEST FIRST WIDTH
    with torch.no_grad():
        Decoder1 = Decoder1.cuda()
        Encoder4.eval()
        Decoder1.eval()
        Encoder_AQL1.eval()
        Decoder_IAQL1.eval()
        priorEncoder_IAQL1.eval()
        priorDecoder_AQL1.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        first_recon_image = []
        first_psnr = []
        first_msssim = []
        for batch_idx, input in enumerate(test_loader):
            if batch_idx > 1000:
                break
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input, (Encoder_AQL1, Decoder_IAQL1, priorEncoder_IAQL1, priorDecoder_AQL1))
            compressed_feature_renorm = Decoder_IAQL1(compressed_feature_renorm)
            recon_image = Decoder1(compressed_feature_renorm)
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            first_psnr.append(psnr)
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            first_msssim.append(msssimDB)
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("First Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))

    # TEST SECOND WIDTH
    with torch.no_grad():
        Decoder2 = Decoder2.cuda()
        gate21 = gate21.cuda()
        Decoder2.eval()
        gate21.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        second_recon_image = []
        second_psnr = []
        second_msssim = []
        for batch_idx, input in enumerate(test_loader):
            if batch_idx > 1000:
                break
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input, (Encoder_AQL1, Decoder_IAQL1, priorEncoder_IAQL1, priorDecoder_AQL1))
            compressed_feature_renorm = Decoder_IAQL1(compressed_feature_renorm)
            recon_image1 = Decoder1(compressed_feature_renorm)
            recon_image2 = Decoder2(compressed_feature_renorm)
            recon_image1 = gate21(recon_image1)
            recon_image = recon_image1 + recon_image2
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            second_psnr.append(psnr)
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            second_msssim.append(msssimDB)
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("Second Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        gate21 = gate21.cpu()

    # TEST THIRD WIDTH
    with torch.no_grad():
        Decoder3 = Decoder3.cuda()
        gate31 = gate31.cuda()
        gate32 = gate32.cuda()
        Decoder3.eval()
        gate31.eval()
        gate32.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            if batch_idx > 1000:
                break
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input, (Encoder_AQL1, Decoder_IAQL1, priorEncoder_IAQL1, priorDecoder_AQL1))
            compressed_feature_renorm = Decoder_IAQL1(compressed_feature_renorm)
            recon_image1 = Decoder1(compressed_feature_renorm)
            recon_image2 = Decoder2(compressed_feature_renorm)
            recon_image3 = Decoder3(compressed_feature_renorm)
            recon_image1 = gate31(recon_image1)
            recon_image2 = gate32(recon_image2)
            recon_image = recon_image1 + recon_image2 + recon_image3
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            psnr_diff = psnr - first_psnr[batch_idx]
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("Third Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        gate31 = gate31.cpu()
        gate32 = gate32.cpu()
    Encoder4.cpu()
    Decoder1.cpu() 
    Decoder2.cpu() 
    Decoder3.cpu() 
    Encoder_AQL1.cpu()
    Decoder_IAQL1.cpu()
    priorEncoder_IAQL1.cpu()
    priorDecoder_AQL1.cpu()

    # TEST SECOND BPP
    Encoder4 = Encoder4.cuda()
    Encoder_AQL2.cuda()
    Decoder_IAQL2.cuda()
    priorEncoder_IAQL2.cuda()
    priorDecoder_AQL2.cuda()
    # TEST FIRST WIDTH
    with torch.no_grad():
        Decoder1 = Decoder1.cuda()
        Encoder4.eval()
        Decoder1.eval()
        Encoder_AQL2.eval()
        Decoder_IAQL2.eval()
        priorEncoder_IAQL2.eval()
        priorDecoder_AQL2.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        first_recon_image = []
        first_psnr = []
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input, (Encoder_AQL2, Decoder_IAQL2, priorEncoder_IAQL2, priorDecoder_AQL2))
            compressed_feature_renorm = Decoder_IAQL2(compressed_feature_renorm)
            recon_image = Decoder1(compressed_feature_renorm)
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("First Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))

    # TEST SECOND WIDTH
    with torch.no_grad():
        Decoder2 = Decoder2.cuda()
        gate21 = gate21.cuda()
        Decoder2.eval()
        gate21.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input, (Encoder_AQL2, Decoder_IAQL2, priorEncoder_IAQL2, priorDecoder_AQL2))
            compressed_feature_renorm = Decoder_IAQL2(compressed_feature_renorm)
            recon_image1 = Decoder1(compressed_feature_renorm)
            recon_image2 = Decoder2(compressed_feature_renorm)
            recon_image1 = gate21(recon_image1)
            recon_image = recon_image1 + recon_image2
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("Second Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        gate21 = gate21.cpu()

    # TEST THIRD WIDTH
    with torch.no_grad():
        Decoder3 = Decoder3.cuda()
        gate31 = gate31.cuda()
        gate32 = gate32.cuda()
        Decoder3.eval()
        gate31.eval()
        gate32.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input, (Encoder_AQL2, Decoder_IAQL2, priorEncoder_IAQL2, priorDecoder_AQL2))
            compressed_feature_renorm = Decoder_IAQL2(compressed_feature_renorm)
            recon_image1 = Decoder1(compressed_feature_renorm)
            recon_image2 = Decoder2(compressed_feature_renorm)
            recon_image3 = Decoder3(compressed_feature_renorm)
            recon_image1 = gate31(recon_image1)
            recon_image2 = gate32(recon_image2)
            recon_image = recon_image1 + recon_image2 + recon_image3
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("Third Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        gate31 = gate31.cpu()
        gate32 = gate32.cpu()
    Encoder4.cpu()
    Decoder1.cpu() 
    Decoder2.cpu() 
    Decoder3.cpu() 
    Encoder_AQL2.cpu()
    Decoder_IAQL2.cpu()
    priorEncoder_IAQL2.cpu()
    priorDecoder_AQL2.cpu()

    # TEST THIRD BPP
    Encoder4 = Encoder4.cuda()
    Encoder_AQL3.cuda()
    Decoder_IAQL3.cuda()
    priorEncoder_IAQL3.cuda()
    priorDecoder_AQL3.cuda()
    # TEST FIRST WIDTH
    with torch.no_grad():
        Decoder1 = Decoder1.cuda()
        Encoder4.eval()
        Decoder1.eval()
        Encoder_AQL3.eval()
        Decoder_IAQL3.eval()
        priorEncoder_IAQL3.eval()
        priorDecoder_AQL3.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        first_recon_image = []
        first_psnr = []
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input, (Encoder_AQL3, Decoder_IAQL3, priorEncoder_IAQL3, priorDecoder_AQL3))
            compressed_feature_renorm = Decoder_IAQL3(compressed_feature_renorm)
            recon_image = Decoder1(compressed_feature_renorm)
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("First Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))

    # TEST SECOND WIDTH
    with torch.no_grad():
        Decoder2 = Decoder2.cuda()
        gate21 = gate21.cuda()
        Decoder2.eval()
        gate21.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input, (Encoder_AQL3, Decoder_IAQL3, priorEncoder_IAQL3, priorDecoder_AQL3))
            compressed_feature_renorm = Decoder_IAQL3(compressed_feature_renorm)
            recon_image1 = Decoder1(compressed_feature_renorm)
            recon_image2 = Decoder2(compressed_feature_renorm)
            recon_image1 = gate21(recon_image1)
            recon_image = recon_image1 + recon_image2
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("Second Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        gate21 = gate21.cpu()

    # TEST THIRD WIDTH
    with torch.no_grad():
        Decoder3 = Decoder3.cuda()
        gate31 = gate31.cuda()
        gate32 = gate32.cuda()
        Decoder3.eval()
        gate31.eval()
        gate32.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input, (Encoder_AQL3, Decoder_IAQL3, priorEncoder_IAQL3, priorDecoder_AQL3))
            compressed_feature_renorm = Decoder_IAQL3(compressed_feature_renorm)
            recon_image1 = Decoder1(compressed_feature_renorm)
            recon_image2 = Decoder2(compressed_feature_renorm)
            recon_image3 = Decoder3(compressed_feature_renorm)
            recon_image1 = gate31(recon_image1)
            recon_image2 = gate32(recon_image2)
            recon_image = recon_image1 + recon_image2 + recon_image3
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("Third Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        gate31 = gate31.cpu()
        gate32 = gate32.cpu()
    Encoder4.cpu()
    Decoder1.cpu() 
    Decoder2.cpu() 
    Decoder3.cpu() 
    Encoder_AQL3.cpu()
    Decoder_IAQL3.cpu()
    priorEncoder_IAQL3.cpu()
    priorDecoder_AQL3.cpu()

    # TEST FOURTH BPP
    Encoder4 = Encoder4.cuda()
    # TEST FIRST WIDTH
    with torch.no_grad():
        Decoder1 = Decoder1.cuda()
        Encoder4.eval()
        Decoder1.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        first_recon_image = []
        first_psnr = []
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input)
            recon_image = Decoder1(compressed_feature_renorm)
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("First Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))

    # TEST SECOND WIDTH
    with torch.no_grad():
        Decoder2 = Decoder2.cuda()
        gate21 = gate21.cuda()
        Decoder2.eval()
        gate21.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input)
            recon_image1 = Decoder1(compressed_feature_renorm)
            recon_image2 = Decoder2(compressed_feature_renorm)
            recon_image1 = gate21(recon_image1)
            recon_image = recon_image1 + recon_image2
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("Second Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        gate21 = gate21.cpu()

    # TEST THIRD WIDTH
    with torch.no_grad():
        Decoder3 = Decoder3.cuda()
        gate31 = gate31.cuda()
        gate32 = gate32.cuda()
        Decoder3.eval()
        gate31.eval()
        gate32.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            bpp_feature, bpp_z, bpp, compressed_feature_renorm = Encoder4(input)
            recon_image1 = Decoder1(compressed_feature_renorm)
            recon_image2 = Decoder2(compressed_feature_renorm)
            recon_image3 = Decoder3(compressed_feature_renorm)
            recon_image1 = gate31(recon_image1)
            recon_image2 = gate32(recon_image2)
            recon_image = recon_image1 + recon_image2 + recon_image3
            mse_loss = torch.mean((recon_image - input).pow(2))
            clipped_recon_image = recon_image.clamp(0., 1.)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        print("Third Width Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        gate31 = gate31.cpu()
        gate32 = gate32.cpu()
    Encoder4.cpu()
    Decoder1.cpu() 
    Decoder2.cpu() 
    Decoder3.cpu() 

if __name__ == "__main__":
    args = parser.parse_args()
    config = parse_config(args)
    testKodak(args, config)
    
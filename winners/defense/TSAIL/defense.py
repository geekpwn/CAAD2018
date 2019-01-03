"""Sample Pytorch defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import math
import numpy as np
import time

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

from dataset import Dataset
from res152_wide import get_model as get_model1
from inres import get_model as  get_model2
from v3 import get_model as get_model3
from resnext101 import get_model as get_model4

begin_time = time.time()

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR', default='',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE', default='',
                    help='Output file to save labels.')
parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='Batch size (default: 8)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')


parser.add_argument('--random_per_sample', type=int, default=1,
                    help='num of random teacher for each sample')
parser.add_argument('--shrink', type=float, default=1.0,
                    help='ratio of shrink original images')
parser.add_argument('--noise', type=float, default=0.5,
                    help='ratio of random teacher')

class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor



def main():
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print("Error: Invalid input folder %s" % args.input_dir)
        exit(-1)
    if not args.output_file:
        print("Error: Please specify an output file")
        exit(-1)

    tf = transforms.Compose([
           transforms.Scale([299,299]),
            transforms.ToTensor()
    ])

    mean_torch = autograd.Variable(torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    std_torch = autograd.Variable(torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    mean_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    std_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)


    dataset = Dataset(args.input_dir, transform=tf)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    config, resmodel = get_model1(args.shrink,args.noise)
    config, inresmodel = get_model2(args.shrink,args.noise)
    config, incepv3model = get_model3(args.shrink,args.noise)
    config, rexmodel = get_model4(args.shrink,args.noise)
    net1 = resmodel.net
    net2 = inresmodel.net
    net3 = incepv3model.net
    net4 = rexmodel.net

    checkpoint = torch.load('denoise_res.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        resmodel.load_state_dict(checkpoint['state_dict'])
    else:
        resmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_inres.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        inresmodel.load_state_dict(checkpoint['state_dict'])
    else:
        inresmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_incep.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        incepv3model.load_state_dict(checkpoint['state_dict'])
    else:
        incepv3model.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_rex.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        rexmodel.load_state_dict(checkpoint['state_dict'])
    else:
        rexmodel.load_state_dict(checkpoint)

    if not args.no_gpu:
        inresmodel = inresmodel.cuda()
        resmodel = resmodel.cuda()
        incepv3model = incepv3model.cuda()
        rexmodel = rexmodel.cuda()
    inresmodel.eval()
    resmodel.eval()
    incepv3model.eval()
    rexmodel.eval()

    outputs = []
    for batch_idx, (input, _) in enumerate(loader):
        if not args.no_gpu:
            input = input.cuda()
        b_size = input.size()[0]
        input_repeat = input.repeat(args.random_per_sample,1,1,1)
        input_var = autograd.Variable(input_repeat, volatile=True)
        input_tf = (input_var-mean_tf)/std_tf
        input_torch = (input_var - mean_torch)/std_torch

        #clean1 = net1.denoise[0](input_torch)
        #clean2 = net2.denoise[0](input_tf)
        #clean3 = net3.denoise(input_tf)

        #labels1 = net1(clean1,False)[-1]
        #labels2 = net2(clean2,False)[-1]
        #labels3 = net3(clean3,False)[-1]

        labels1 = net1(input_torch,True)[-1]
        labels2 = net2(input_tf,True)[-1]
        labels3 = net3(input_tf,True)[-1]
        labels4 = net4(input_torch,True)[-1]

        labels1_n = np.reshape(labels1.data.cpu().numpy(),(-1, b_size, 1000))
        labels2_n = np.reshape(labels2.data.cpu().numpy(),(-1, b_size, 1000))
        labels3_n = np.reshape(labels3.data.cpu().numpy(),(-1, b_size, 1000))
        labels4_n = np.reshape(labels4.data.cpu().numpy(),(-1, b_size, 1000))

        labels1_s = np.sum(labels1_n,axis=0)
        labels2_s = np.sum(labels2_n,axis=0)
        labels3_s = np.sum(labels3_n,axis=0)
        labels4_s = np.sum(labels4_n,axis=0)

        labels = np.argmax(labels1_s+labels2_s+labels3_s+labels4_s,axis=1) + 1  # argmax + offset to match Google's Tensorflow + Inception 1001 class ids
        outputs.append(labels)
    outputs = np.concatenate(outputs, axis=0)

    with open(args.output_file, 'w') as out_file:
        filenames = dataset.filenames()
        for filename, label in zip(filenames, outputs):
            filename = os.path.basename(filename)
            out_file.write('{0},{1}\n'.format(filename, label))

    print('Time cost is %f s'%(time.time()-begin_time))

if __name__ == '__main__':
    main()

import argparse
import os
import time
import pickle

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from model.dataset import GraspCustomLabelDataset
from model.gpd import GPDClassifier

parser = argparse.ArgumentParser(description='pointnetGPD')
parser.add_argument('--config', type=str, default='configuration file of the method to be compared')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--load-model', type=str, default='')
parser.add_argument('--load-epoch', type=int, default=-1)
parser.add_argument('--model-path', type=str, default='./assets/learned_models',
                   help='pre-trained model path')
parser.add_argument('--data-path', type=str, default='./data', help='data path')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=1)

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)

logger = SummaryWriter(os.path.join('./assets/log/', args.tag))
np.random.seed(int(time.time()))

def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2**31-1))

def my_collate(batch):
    batch = list(filter(lambda x:x is not None and x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

grasp_points_num=1000
input_size=60 # Only support 60
input_chann=12 # Use 12-channel version for fair comparison

with open(args.config, 'r') as fp:
    config = json.load(fp)

dataset = GraspCustomLabelDataset(
        config,
        grasp_points_num=grasp_points_num,
        projection=True, # for GPD
        project_chann=input_chann, # for GPD {3,12}-channel version
        project_size=input_size,
)
dataset.train()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=16,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)

is_resume = 0
if args.load_model and args.load_epoch != -1:
    is_resume = 1

model = GPDClassifier(input_chann)
if is_resume or args.mode == 'test':
    model.load_state_dict(torch.load(args.load_model))
    print('load weights {}'.format(args.load_model))
if args.cuda:
    base_model = model.cuda()
    model = nn.DataParallel(base_model)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

def train(model, dataset, loader, epoch):
    scheduler.step()
    dataset.train()
    model.train()
    torch.set_grad_enabled(True)
    correct = 0
    dataset_size = 0
    for batch_idx, (data, target) in enumerate(loader):
        dataset_size += data.shape[0]
        data, target = data.float(), target.long() #.squeeze() # Why would you do that?
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(
            epoch, batch_idx * args.batch_size, len(loader.dataset),
            100. * batch_idx * args.batch_size / len(loader.dataset), loss.item(), args.tag))
            logger.add_scalar('train_loss', loss.cpu().item(),
                    batch_idx + epoch * len(loader))
    return float(correct)/float(dataset_size)


def test(model, dataset, loader):
    model.eval()
    dataset.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0
    da = {}
    db = {}
    res = []
    for data, target, obj_name in loader:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long() #.squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data) # N*C
        test_loss += F.nll_loss(output, target, size_average=False).cpu().item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
            res.append((i, j[0], k))

    test_loss /= len(loader.dataset)
    acc = float(correct)/float(dataset_size)
    return acc, test_loss


def main():
    if args.mode == 'train':
        for epoch in range(is_resume*args.load_epoch, args.epoch):
            acc_train = train(model, dataset, dataloader, epoch)
            print('Train done, acc={}'.format(acc_train))
            acc, loss = test(model, dataset, dataloader)
            print('Test done, acc={}, loss={}'.format(acc, loss))
            logger.add_scalar('train_acc', acc_train, epoch)
            logger.add_scalar('test_acc', acc, epoch)
            logger.add_scalar('test_loss', loss, epoch)
            if epoch % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + '_{}.pth'.format(epoch))
                torch.save(base_model.state_dict(), path)
                print('Save weights @ {}'.format(path))
    else:
        print('testing...')
        acc, loss = test(model, test_loader)
        print('Test done, acc={}, loss={}'.format(acc, loss))

if __name__ == "__main__":
    main()

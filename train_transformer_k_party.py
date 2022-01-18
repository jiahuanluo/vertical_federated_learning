import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from models.transformer_k_party import Manual_A, Manual_B
from dataset import Multimodal_Datasets

parser = argparse.ArgumentParser("mosei")
parser.add_argument('--data', required=True, help='location of the data corpus')
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=0, help='num of workers')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--layers', type=int, default=18, help='total number of layers')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=100, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
parser.add_argument('--k', type=int, required=True, help='num of client')
parser.add_argument('--ratio', type=float, default=1.0, help='portion of train samples')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='optimizer')

args = parser.parse_args()

args.name = 'experiments/{}-{}'.format(args.name, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.name, scripts_to_save=glob.glob('*/*.py') + glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
exp_name = f'{args.k}_party_{args.optimizer}_lr{args.learning_rate}_bz{args.batch_size}_epochs_{args.epochs}_ratio_' \
           f'{args.ratio}.txt'
fh = logging.FileHandler(os.path.join(args.name, exp_name))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(args.name, 'tb'))
writer.add_text('expername', args.name, 0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # if not torch.cuda.is_available():
    #     logging.info('no gpu device available')
    #     sys.exit(1)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)
        logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    model_A = Manual_A(num_classes=1, input_dim=35, layers=args.layers, u_dim=30, k=args.k)
    if args.k == 1:
        model_list = [model_A]
    elif args.k == 2:
        model_B = Manual_B(input_dim=300)
        model_list = [model_A, model_B]
    elif args.k == 3:
        model_B = Manual_B(input_dim=300)
        model_C = Manual_B(input_dim=74)
        model_list = [model_A, model_B, model_C]
    else:
        assert ValueError
    # model_list = [model_A] + [Manual_B(layers=args.layers, u_dim=args.u_dim) for _ in range(args.k - 1)]
    model_list = [model.to(device) for model in model_list]

    for i in range(args.k):
        logging.info("model_{} param size = {}MB".format(i + 1, utils.count_parameters_in_MB(model_list[i])))

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.L1Loss()
    if args.optimizer == "sgd":
        optimizer_list = [torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                          weight_decay=args.weight_decay) for model in model_list]
    else:
        optimizer_list = [torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay) for model in model_list]
    train_data = Multimodal_Datasets(args.data, split_type='train', ratio=args.ratio)
    valid_data = Multimodal_Datasets(args.data, split_type='test')
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

    if args.learning_rate == 0.025:
        scheduler_list = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
            for optimizer in optimizer_list]
    else:
        scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma) for optimizer
                          in optimizer_list]

    best_f1 = 0
    best_acc = 0
    for epoch in range(args.epochs):
        lr = scheduler_list[0].get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        cur_step = epoch * len(train_queue)
        writer.add_scalar('train/lr', lr, cur_step)

        train_f1, train_obj = train(train_queue, model_list, criterion, optimizer_list, epoch)
        [scheduler_list[i].step() for i in range(len(scheduler_list))]
        logging.info('train_f1 %f', train_f1)

        cur_step = (epoch + 1) * len(train_queue)

        valid_f1, valid_acc, valid_obj = infer(valid_queue, model_list, criterion, epoch, cur_step)
        logging.info('valid_f1 %f', valid_f1)
        logging.info('valid_acc %f', valid_acc)

        # transfer_valid_acc_top1, transfer_valid_acc_top5, transfer_valid_obj = transfer_infer(valid_queue, model_list,
        #                                                                                       criterion, epoch,
        #                                                                                       cur_step)
        # logging.info('transfer_valid_acc_top1 %f', transfer_valid_acc_top1)
        # logging.info('transfer_valid_acc_top5 %f', transfer_valid_acc_top5)

        if valid_f1 > best_f1:
            best_f1 = valid_f1
        if valid_acc > best_acc:
            best_acc = valid_acc
        logging.info('best_f1 %f', best_f1)
        logging.info('best_acc %f', best_acc)


def train(train_queue, model_list, criterion, optimizer_list, epoch):
    objs = utils.AvgrageMeter()
    f1 = utils.AvgrageMeter()
    acc = utils.AvgrageMeter()

    cur_step = epoch * len(train_queue)

    model_list = [model.train() for model in model_list]
    k = len(model_list)

    for step, (trn_X, trn_y) in enumerate(train_queue):
        trn_X = [x.float().to(device) for x in trn_X]
        target = trn_y.view(-1).long().to(device)
        target = target.unsqueeze(1)
        n = target.size(0)
        [optimizer_list[i].zero_grad() for i in range(k)]
        U_B_list = None
        U_B_clone_list = None
        if k > 1:
            U_B_list = [model_list[i](trn_X[i]) for i in range(1, len(model_list))]
            U_B_clone_list = [U_B.detach().clone() for U_B in U_B_list]
            U_B_clone_list = [torch.autograd.Variable(U_B, requires_grad=True) for U_B in U_B_clone_list]
        logits = model_list[0](trn_X[0], U_B_clone_list)
        loss = criterion(logits, target)
        if k > 1:
            U_B_gradients_list = [torch.autograd.grad(loss, U_B, retain_graph=True) for U_B in U_B_clone_list]
            model_B_weights_gradients_list = [
                torch.autograd.grad(U_B_list[i], model_list[i + 1].parameters(), grad_outputs=U_B_gradients_list[i],
                                    retain_graph=True) for i in range(len(U_B_gradients_list))]
            for i in range(len(model_B_weights_gradients_list)):
                for w, g in zip(model_list[i + 1].parameters(), model_B_weights_gradients_list[i]):
                    w.grad = g.detach()
                nn.utils.clip_grad_norm_(model_list[i + 1].parameters(), args.grad_clip)
                optimizer_list[i + 1].step()
        loss.backward()
        nn.utils.clip_grad_norm_(model_list[0].parameters(), args.grad_clip)
        optimizer_list[0].step()

        prec1, prec5 = utils.eval_mosei_senti(results=logits, truths=target, exclude_zero=True)
        objs.update(loss.item(), n)
        f1.update(prec1.item(), n)
        acc.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({f1.avg:.1f}%, {acc.avg:.1f}%)".format(
                    epoch + 1, args.epochs, step, len(train_queue) - 1, losses=objs,
                    f1=f1, acc=acc))
        writer.add_scalar('train/loss', objs.avg, cur_step)
        writer.add_scalar('train/f1', f1.avg, cur_step)
        writer.add_scalar('train/acc', acc.avg, cur_step)
        cur_step += 1
    return f1.avg, objs.avg


def infer(valid_queue, model_list, criterion, epoch, cur_step):
    objs = utils.AvgrageMeter()
    f1 = utils.AvgrageMeter()
    acc = utils.AvgrageMeter()
    model_list = [model.eval() for model in model_list]
    k = len(model_list)
    with torch.no_grad():
        for step, (val_X, val_y) in enumerate(valid_queue):
            val_X = [x.float().to(device) for x in val_X]
            target = val_y.view(-1).long().to(device)
            target = target.unsqueeze(1)
            n = target.size(0)
            U_B_list = None
            if k > 1:
                U_B_list = [model_list[i](val_X[i]) for i in range(1, len(model_list))]
            logits = model_list[0](val_X[0], U_B_list)
            loss = criterion(logits, target)
            prec1, prec5 = utils.eval_mosei_senti(results=logits, truths=target, exclude_zero=True)
            objs.update(loss.item(), n)
            f1.update(prec1.item(), n)
            acc.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({f1.avg:.1f}%, {acc.avg:.1f}%)".format(
                        epoch + 1, args.epochs, step, len(valid_queue) - 1, losses=objs,
                        f1=f1, acc=acc))
    writer.add_scalar('valid/loss', objs.avg, cur_step)
    writer.add_scalar('valid/f1', f1.avg, cur_step)
    writer.add_scalar('valid/acc', acc.avg, cur_step)
    return f1.avg, acc.avg, objs.avg


if __name__ == '__main__':
    main()

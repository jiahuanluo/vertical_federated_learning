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
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--workers', type=int, default=0, help='num of workers')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--layers', type=int, default=18, help='total number of layers')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=0.8, help='gradient clipping')
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
        # torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)
        logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    model_A = Manual_A(num_classes=1, input_dim=74, layers=args.layers, u_dim=30, k=args.k)
    if args.k == 1:
        model_list = [model_A]
    elif args.k == 2:
        model_B = Manual_B(input_dim=300)
        model_list = [model_A, model_B]
    elif args.k == 3:
        model_B = Manual_B(input_dim=300)
        model_C = Manual_B(input_dim=35)
        model_list = [model_A, model_B, model_C]
    else:
        assert ValueError
    # model_list = [model_A] + [Manual_B(layers=args.layers, u_dim=args.u_dim) for _ in range(args.k - 1)]
    model_list = [model.to(device) for model in model_list]

    for i in range(args.k):
        logging.info("model_{} param size = {}MB".format(i + 1, utils.count_parameters_in_MB(model_list[i])))

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.L1Loss()
    optimizer_list = [torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay) for model in model_list]
    train_data = Multimodal_Datasets(args.data, split_type='train', ratio=args.ratio)
    valid_data = Multimodal_Datasets(args.data, split_type='test')
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    scheduler_list = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1, verbose=True) for
        optimizer in optimizer_list]

    best_f1 = 0
    best_acc = 0
    for epoch in range(args.epochs):
        # lr = scheduler_list[0].get_last_lr()[0]
        lr = 1e-3
        logging.info('epoch %d lr %e', epoch, lr)

        cur_step = epoch * len(train_queue)
        writer.add_scalar('train/lr', lr, cur_step)

        data_dict = train(train_queue, model_list, criterion, optimizer_list, epoch)
        logging.info(f'train_data_dict: {data_dict}')
        for key, value in data_dict.items():
            writer.add_scalar(f"train/{key}", value, cur_step)
        cur_step = (epoch + 1) * len(train_queue)
        [scheduler_list[i].step(data_dict['avg_loss']) for i in range(len(scheduler_list))]

        data_dict = infer(valid_queue, model_list, criterion, epoch, cur_step)
        logging.info(f'valid_data_dict: {data_dict}')
        for key, value in data_dict.items():
            writer.add_scalar(f"valid/{key}", value, cur_step)

        # transfer_valid_acc_top1, transfer_valid_acc_top5, transfer_valid_obj = transfer_infer(valid_queue, model_list,
        #                                                                                       criterion, epoch,
        #                                                                                       cur_step)
        # logging.info('transfer_valid_acc_top1 %f', transfer_valid_acc_top1)
        # logging.info('transfer_valid_acc_top5 %f', transfer_valid_acc_top5)

        if data_dict['f_score'] > best_f1:
            best_f1 = data_dict['f_score']
        if data_dict['acc'] > best_acc:
            best_acc = data_dict['acc']
        logging.info('best_f1 %f', best_f1)
        logging.info('best_acc %f', best_acc)


def train(train_queue, model_list, criterion, optimizer_list, epoch):
    objs = utils.AvgrageMeter()
    cur_step = epoch * len(train_queue)

    model_list = [model.train() for model in model_list]
    k = len(model_list)
    results = []
    truths = []
    for step, (trn_X, trn_y) in enumerate(train_queue):
        trn_X = [x.float().to(device) for x in trn_X]
        target = trn_y.view(-1).long().to(device)
        target = target.unsqueeze(1)
        n = target.size(0)
        [model_list[i].zero_grad() for i in range(k)]
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
        results.append(logits)
        truths.append(target)
        objs.update(loss.item(), n)
        if step % args.report_freq == 0:
            logging.info(
                f"Train: [{epoch + 1:2d}/{args.epochs}] Step {step:03d}/{len(train_queue) - 1:03d} Loss {objs.avg:.3f} ")
        writer.add_scalar('train/loss', objs.avg, cur_step)
        cur_step += 1
    results = torch.cat(results)
    truths = torch.cat(truths)
    data_dict = utils.eval_mosei_senti(results=results, truths=truths, exclude_zero=True)
    data_dict['avg_loss'] = objs.avg
    return data_dict


def infer(valid_queue, model_list, criterion, epoch, cur_step):
    objs = utils.AvgrageMeter()
    model_list = [model.eval() for model in model_list]
    k = len(model_list)
    results = []
    truths = []
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
            results.append(logits)
            truths.append(target)
            objs.update(loss.item(), n)
            if step % args.report_freq == 0:
                logging.info(
                    f"Valid: [{epoch + 1:2d}/{args.epochs}] Step {step:03d}/{len(valid_queue) - 1:03d} Loss {objs.avg:.3f}")
            writer.add_scalar('valid/loss', objs.avg, cur_step)
            cur_step += 1
    results = torch.cat(results)
    truths = torch.cat(truths)
    data_dict = utils.eval_mosei_senti(results=results, truths=truths, exclude_zero=True)
    data_dict['avg_loss'] = objs.avg
    return data_dict


if __name__ == '__main__':
    main()

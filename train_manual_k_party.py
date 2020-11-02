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
from models.manual_k_party import Manual_A, Manual_B
from dataset import NUS_WIDE_2_Party
from sklearn import metrics

parser = argparse.ArgumentParser("NUS_WIDE_2_Party")
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--batch_size', type=int, default=48, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--layers', type=int, default=50, help='total number of layers')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
parser.add_argument('--k', type=int, required=True, help='num of client')

args = parser.parse_args()

args.name = 'eval/{}-{}'.format(args.name, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.name, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.name, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(args.name, 'tb'))
writer.add_text('expername', args.name, 0)

class_label_list = ['person', 'animal']


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed_all(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    model_A = Manual_A(634, u_dim=args.u_dim, k=args.k)
    model_list = [model_A] + [Manual_B(1000, u_dim=args.u_dim) for _ in range(args.k - 1)]
    model_list = [model.cuda() for model in model_list]

    for i in range(args.k):
        logging.info("model_{} param size = {}MB".format(i + 1, utils.count_parameters_in_MB(model_list[i])))

    criterion = nn.BCEWithLogitsLoss()
    optimizer_list = [
        torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        for model in model_list]
    dataset = NUS_WIDE_2_Party(args.data, class_label_list, 'Train', 2)
    num_train = len(dataset)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(0.8 * num_train))
    train_queue = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False,
        num_workers=0
    )

    valid_queue = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=False,
        num_workers=0
    )
    if args.learning_rate == 0.025:
        scheduler_list = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
            for optimizer in optimizer_list]
    else:
        scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma) for optimizer
                          in optimizer_list]
    best_acc = 0
    best_auc = 0
    for epoch in range(args.epochs):
        [scheduler_list[i].step() for i in range(len(scheduler_list))]
        lr = scheduler_list[0].get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        cur_step = epoch * len(train_queue)
        writer.add_scalar('train/lr', lr, cur_step)

        cur_step = epoch * len(train_queue)
        writer.add_scalar('train/lr', lr, cur_step)

        train_acc, train_obj = train(train_queue, model_list, criterion, optimizer_list, epoch)
        logging.info('train_acc %f', train_acc)

        cur_step = (epoch + 1) * len(train_queue)
        valid_acc, valid_obj, valid_auc = infer(valid_queue, model_list, criterion, epoch, cur_step)

        logging.info('Valid_acc %f', valid_acc)
        logging.info('Valid_auc %f', valid_auc)

        if valid_acc > best_acc:
            best_acc = valid_acc
        if valid_auc > best_auc:
            best_auc = valid_auc
        logging.info('Best_Valid_acc %f', best_acc)
        logging.info('Best_Valid_auc %f', best_auc)


def train(train_queue, model_list, criterion, optimizer_list, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model_list = [model.train() for model in model_list]
    cur_step = epoch * len(train_queue)
    k = len(model_list)
    for step, (trn_X, trn_y) in enumerate(train_queue):
        trn_X = [x.float().cuda() for x in trn_X]
        target = trn_y.float().cuda()
        n = target.size(0)
        [optimizer_list[i].zero_grad() for i in range(k)]
        U_B_list = None
        if k > 1:
            U_B_list = [model_list[i](trn_X[i]) for i in range(1, len(model_list))]
        logits, dist_loss = model_list[0](trn_X[0], U_B_list)
        loss = criterion(logits.view(-1), target.view(-1))
        loss = loss + 0.5 * dist_loss
        if k > 1:
            U_B_gradients_list = [torch.autograd.grad(loss, U_B, retain_graph=True) for U_B in U_B_list]
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

        prec1, _ = utils.accuracy(logits, target)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1) ({top1.avg:.1f}%)".format(
                    epoch + 1, args.epochs, step, len(train_queue) - 1, losses=objs,
                    top1=top1))
        writer.add_scalar('train/loss', objs.avg, cur_step)
        writer.add_scalar('train/top1', top1.avg, cur_step)
        cur_step += 1

    return top1.avg, objs.avg


def infer(valid_queue, model_list, criterion, epoch, cur_step):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model_list = [model.eval() for model in model_list]
    k = len(model_list)
    pred_list = []
    true_list = []
    prob_list = []
    with torch.no_grad():
        for step, (val_X, val_y) in enumerate(valid_queue):
            val_X = [x.float().cuda() for x in val_X]
            target = val_y.float().cuda()
            n = target.size(0)
            U_B_list = None
            if k > 1:
                U_B_list = [model_list[i](val_X[i]) for i in range(1, len(model_list))]
            logits, _ = model_list[0](val_X[0], U_B_list)
            loss = criterion(logits.view(-1), target.view(-1))
            prec1, label = utils.accuracy(logits, target)
            objs.update(loss.item(), n)
            top1.update(prec1, n)
            pred_list.extend(label.view(-1).cpu().detach().tolist())
            prob_list.extend(logits.view(-1).cpu().detach().tolist())
            true_list.extend(target.view(-1).cpu().detach().tolist())
            if step % args.report_freq == 0:
                logging.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1f}%)".format(
                        epoch + 1, args.epochs, step, len(valid_queue) - 1, losses=objs,
                        top1=top1))
    auc = metrics.roc_auc_score(true_list, prob_list) * 100
    logging.info(
        "Valid: [{:2d}/{}] Loss {losses.avg:.3f} "
        "Prec@(1) ({top1.avg:.1f}%) AUC ({auc:.1f}%)".format(
            epoch + 1, args.epochs, len(valid_queue) - 1, losses=objs,
            top1=top1, auc=auc))
    writer.add_scalar('valid/loss', objs.avg, cur_step)
    writer.add_scalar('valid/top1', top1.avg, cur_step)
    writer.add_scalar('valid/auc', auc, cur_step)
    return top1.avg, objs.avg, auc


if __name__ == '__main__':
    main()

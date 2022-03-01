import subprocess
import time
import os
import logging
from multiprocessing import Process, Queue

logging.basicConfig()
logger = logging.getLogger('pipeline')
fh = logging.FileHandler('pipeline.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)
#####
num_gpus = 3


#####

class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def get_current_time(self):
        return (time.time() - self.start_time) / 3600


class GPUManager(object):
    def __init__(self, num_gpus=4):
        self.gpu_queue = Queue()
        for device_id in range(num_gpus):
            self.gpu_queue.put(device_id)

    def require(self):
        try:
            return self.gpu_queue.get()
        except:
            return None

    def add_gpu(self, gpu_id):
        self.gpu_queue.put(gpu_id)


timer = Timer()
gpu_manager = GPUManager(num_gpus=num_gpus)


def run_gpu_model(cmd, log_file=None):
    if log_file:
        cmd = f"nohup {cmd} > {log_file}"
    while True:
        gpu_id = gpu_manager.require()
        if gpu_id is not None:
            try:
                run_cmd = f"export CUDA_VISIBLE_DEVICES={gpu_id} && {cmd}"
                logger.info(f"{run_cmd} 开始时间: {timer.get_current_time()}")
                os.system(run_cmd)
                logger.info(f"{run_cmd} 结束时间: {timer.get_current_time()}")
            except:
                logger.warning(f'{cmd} failed')
            gpu_manager.add_gpu(gpu_id)
            break


def train_vanilla_classification(k, finetune_ratio, eval_cls_lr):
    logger.info(f"训练分类器 {k}_{finetune_ratio}-{eval_cls_lr} 开始: {timer.get_current_time()}")
    run_gpu_model(
        f'python train_transformer_k_party.py --data ./data --name mosei_{k}_party_ratio_{finetune_ratio} --k {k} --batch_size 128 --gpu 0 --ratio {finetune_ratio}',
        log_file=f"{k}_{finetune_ratio}-{eval_cls_lr}.log")
    logger.info(f"训练分类器 {k}_{finetune_ratio}-{eval_cls_lr} 结束: {timer.get_current_time()}")


model_processes = []
for k in range(1, 3):
    for finetune_ratio in [1, 0.5, 0.2, 0.1, 0.05, 0.01]:
        for eval_cls_lr in [0.001]:
            p = Process(target=train_vanilla_classification,
                        args=(k, finetune_ratio, eval_cls_lr))
            p.start()
            time.sleep(10)
            logger.info(f'创建模型训练 {k}-{finetune_ratio}-{eval_cls_lr}...')
            model_processes.append(p)
for p in model_processes:
    p.join()
logger.info(f'训练vanilla模型结束: {timer.get_current_time()} 小时')

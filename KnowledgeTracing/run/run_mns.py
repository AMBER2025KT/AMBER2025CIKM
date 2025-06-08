from min_norm_solvers import MinNormSolver
import sys
project_root = "/root/KT/AMBER" # replace to your project root
sys.path.append(project_root)
from KnowledgeTracing.DirectedGCN.load_data import get_adj
from KnowledgeTracing.hgnn_models import hypergraph_utils as hgut
from KnowledgeTracing.model.Modelnew import DKT
from KnowledgeTracing.data.dataloader import getLoader
from KnowledgeTracing.Constant import Constants as C
from torch import optim as optima
from KnowledgeTracing.evaluation import eval
from KnowledgeTracing.run.test import test_epoch
import torch
import logging
from datetime import datetime
import numpy as np
import warnings
import os
import random
import pandas as pd
from tqdm import tqdm
from copy import deepcopy as cp
from collections import OrderedDict
from KnowledgeTracing.run.comloss import CombinedLossI, CombinedLossID
from torch.optim.lr_scheduler import LambdaLR

warnings.filterwarnings('ignore')

torch.cuda.set_device(2)

'''check cuda'''
use_gpu = torch.cuda.is_available()
device = torch.device('cuda')
print('GPU state: ', use_gpu)
print('Dataset: ' + C.DATASET + ', Ques number: ' + str(C.NUM_OF_QUESTIONS) + '\n')

''' save log '''
logger = logging.getLogger('main')
logger.setLevel(level=logging.DEBUG)
date = datetime.now()
handler = logging.FileHandler(
    f'log/{date.year}_{date.month}_{date.day}_result2s.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('This is a new training log')
logger.info('\nDataset: ' + str(C.DATASET) + ', Ques number: ' + str(C.NUM_OF_QUESTIONS) + ', Batch_size: ' + str(
    C.BATCH_SIZE))

'''set random seed'''

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(216)

trainLoaders, testLoaders, heldLoaders = getLoader(C.DATASET)




n = 3 # Replace this with the desired epoch threshold


def KTtrain():
    torch.autograd.set_detect_anomaly(True)

    adj = hgut.generate_G_from_H(pd.read_csv(r'../../Dataset/H/' + C.H + '.csv', header=None))
    G = adj.cuda()
    adj_out, adj_in = get_adj()
    adj_in = adj_in.cuda()
    adj_out = adj_out.cuda()

    student_model = DKT(C.HIDDEN, C.LAYERS, G, adj_out, adj_in).cuda()
    teacher_model = DKT(C.HIDDEN, C.LAYERS, G, adj_out, adj_in).cuda()
    
    teacher_path = '/root/KT/AMBER/KnowledgeTracing/model/save2017modelS_weights_teacher.pth' # replace to your teacher model path
    
    teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))

    comblossd = CombinedLossID(device=device)
    comblossc = CombinedLossI(device=device)

    forwardloss = eval.lossFunc(C.HIDDEN, C.MAX_STEP, device)

    student_optimizer = optima.Adam(student_model.parameters(), lr=C.LR)

    teacher_optimizer = optima.Adam(teacher_model.parameters(), lr=C.teacher_learning_rate)

    def linear_decay(step):
        return max(0.0, 1 - step / num_training_steps)

    num_training_steps = C.EPOCH * len(trainLoaders) * 10

    student_scheduler = LambdaLR(student_optimizer, lr_lambda=linear_decay)
    teacher_scheduler = LambdaLR(teacher_optimizer, lr_lambda=linear_decay)


    best_auc = 0.0
    best_epoch = 0
    best_acc = 0.0

    for epoch in range(C.EPOCH):



        skip_steps = epoch > n

        batches_buffer = []

        train_progress = tqdm(trainLoaders, desc=f"Training Epoch {epoch + 1}/{C.EPOCH}", leave=False)
        for d_step, d_batch in enumerate(train_progress):

            student_model_backup_state = cp(student_model.state_dict())
            student_optimizer_backup_state = cp(student_optimizer.state_dict())

            if not skip_steps:
                batches_buffer.append((d_step, d_batch))
                #########################################
                # Step 1: hypothetically update student model #
                #########################################


                fast_weights = OrderedDict(
                    (name, param.clone()) for (name, param) in student_model.named_parameters()
                )

                student_model.train()
                teacher_model.eval()

                for param in teacher_model.parameters():
                    param.requires_grad = True

                assume_loss_total = 0.0
                for step, (buffer_d_step, buffer_d_batch) in enumerate(batches_buffer):
                    logit_c, logit_t, ensemble_logit,  out_h_s, out_d_s = student_model(buffer_d_batch)

                    with torch.no_grad():
                        logit_c_teacher, logit_t_teacher, ensemble_logit_teacher, out_h_t, out_d_t = teacher_model(buffer_d_batch)

                    loss = comblossc(
                        logit_c, logit_t, ensemble_logit,
                        logit_c_teacher, logit_t_teacher, ensemble_logit_teacher,
                        out_h_s, out_d_s, out_h_t, out_d_t, d_batch
                    )



                    grads = torch.autograd.grad(loss, student_model.parameters() if step == 0 else fast_weights.values(),
                                                create_graph=True, allow_unused=True, retain_graph=True)
                    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, fast_weights.values())]

                    fast_weights = OrderedDict(
                        (name, param - C.assume_s_step_size * grad)
                        for ((name, param), grad) in zip(fast_weights.items(), grads)
                    )

                    assume_loss_total += loss.item()

                print(f"\nStep 1: Assume Loss: {assume_loss_total:.4f}")

                #########################################
                # Step 2: use held set to eval s' and update teacher model#
                #########################################

                student_model.eval()
                teacher_model.train()

                for param in teacher_model.parameters():
                    param.requires_grad = True

                with torch.no_grad():
                    for name, param in student_model.named_parameters():
                        param.copy_(fast_weights[name])

                quiz_progress = tqdm(heldLoaders, desc=f"Held Set Epoch {epoch + 1}/{C.EPOCH}", leave=False)
                for q_step, quiz_batch in enumerate(quiz_progress):
                    with torch.no_grad():
                        logit_c_student, logit_t_student, ensemble_logit_student, out_h_s, out_d_s = student_model(quiz_batch)
                    logit_c_teacher, logit_t_teacher, ensemble_logit_teacher, out_h_t, out_d_t = teacher_model(quiz_batch)

                    quiz_loss = comblossc(logit_c_teacher, logit_t_teacher, ensemble_logit_teacher, 
                                          logit_c_student, logit_t_student, ensemble_logit_student,
                                          out_h_s, out_d_s,out_h_t, out_d_t,
                                          quiz_batch
                                         )


                    teacher_grads = torch.autograd.grad(quiz_loss, teacher_model.parameters(), allow_unused=True)
                    teacher_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(teacher_grads, teacher_model.parameters())]

                    for param, grad in zip(teacher_model.parameters(), teacher_grads):
                        param.grad = grad

                    torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm=1.0)

                    teacher_optimizer.step()
                    teacher_optimizer.zero_grad()


                print(f"\nStep 2: Held set loss: {quiz_loss.item():.4f}")
                del teacher_grads
                del grads
                del fast_weights

            #########################################
            # Step 3: update student model#
            #########################################


            del student_model_backup_state, student_optimizer_backup_state

            student_model.train()
            teacher_model.eval()



            logit_c_student, logit_t_student, ensemble_logit_student, out_h_s, out_d_s = student_model(d_batch)
            with torch.no_grad():
                logit_c_teacher, logit_t_teacher, ensemble_logit_teacher, out_h_t, out_d_t = teacher_model(d_batch)

            loss_supervised, loss_kd, loss_embed = comblossd(
                logit_c_student, logit_t_student, ensemble_logit_student,
                logit_c_teacher, logit_t_teacher, ensemble_logit_teacher,
                out_h_s, out_d_s, out_h_t, out_d_t, d_batch
            )

            loss_supervised.backward(retain_graph=True)
            grads_supervised = [param.grad.clone() for param in student_model.parameters() if param.grad is not None]

            student_optimizer.zero_grad()
            loss_kd.backward(retain_graph=True)
            grads_kd = [param.grad.clone() for param in student_model.parameters() if param.grad is not None]

            embed_weight = 1.0

            vecs = [grads_supervised, grads_kd]
            sol, _ = MinNormSolver.find_min_norm_element(vecs)
            supervised_weight, distillation_weight = sol


            total_loss = (supervised_weight * loss_supervised +
                        distillation_weight * loss_kd +
                        embed_weight * loss_embed)

            student_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            student_optimizer.step()


            student_scheduler.step()



            batches_buffer = []

            student_scheduler.step()
            teacher_scheduler.step()

            print(f"\nStep 3: Total loss (student update): {total_loss.item():.4f}")


        with torch.no_grad():
            auc, acc = test_epoch(student_model, testLoaders, device)
            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                best_epoch = epoch + 1
                torch.save(student_model.state_dict(), '../model/save' + C.H + 'modelS_weights.pth')

            
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1} Results Summary")
            logger.info(f"{'='*50}")
        logger.info(f"Overall Performance:")
        logger.info(f"Validation AUC: {auc:.4f}, Accuracy: {acc:.4f}")
        logger.info(f"Best AUC: {best_auc:.4f}, Best Accuracy: {best_acc:.4f}, Best Epoch: {best_epoch}")


def KTtest():
    loss_func = eval.lossFunc(hidden=C.HIDDEN, max_step=C.MAX_STEP, device=torch.device('cuda'))
    adj = hgut.generate_G_from_H(pd.read_csv(r'../../Dataset/H/' + C.H + '.csv', header=None))
    G = adj.cuda()
    adj_out, adj_in = get_adj()
    adj_in = adj_in.cuda()
    adj_out = adj_out.cuda()

    model = DKT(C.HIDDEN, C.LAYERS, G, adj_out, adj_in).cuda()

    model.load_state_dict(torch.load('../model/save' + C.H + 'modelS_weights.pth'))

    model.eval()

    print('loading the best model...')

    with torch.no_grad():
        auc, acc = test_epoch(model, testLoaders, device=torch.device('cuda'))


        print(f"Validation AUC: {auc:.4f}, Accuracy: {acc:.4f}")





KTtrain()
KTtest()
import torch
import wandb
from backbone.backbone import Backbone, AnnealingBackbone
from typing import *
import numpy as np
from utils.logger import print_mean_accuracy
import tqdm
from copy import deepcopy
from utils.sam import SAM
import itertools

def CustomCosineAnnealing(epoch, total_epochs, first_task=False, initial_lr=1.0, min_lr=0.1):
    if first_task:
        # Normal cosine annealing (start high, end low)
        cos_inner = np.pi * epoch / total_epochs
        lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(cos_inner))
    else:
        # Start at min_lr, peak at max_lr at total_epochs//2, and decrease back to min_lr
        if epoch <= total_epochs // 2:
            cos_inner = np.pi * epoch / (total_epochs // 2)
            lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 - np.cos(cos_inner))
        else:
            cos_inner = np.pi * (epoch - total_epochs // 2) / (total_epochs // 2)
            lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(cos_inner))
    
    # Ensure the learning rate does not fall below the minimum learning rate
    lr = max(lr, min_lr)
    return lr


class Trainer():

    def __init__(self, args, model: Backbone, device):

        self.args = args
        self.model = model
        self.device = device

    def ema_update(self, model, ori_model, alpha=0.5):
        for param, ori_param in zip(model.parameters(), ori_model.parameters()):
            param.data = alpha * param.data + (1 - alpha) * ori_param.data

    def end_task(self):

        self.model.end_task()

    def train(self, task_id, task_ids: dict, train_loader, val_loader, test_loader_list, gen_train_loader=None, gen_val_loader=None):

        if self.args.c_sam:
            base_optimizer = torch.optim.SGD
            self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=self.args.c_lr, weight_decay=self.args.c_wd)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.c_lr, weight_decay=self.args.c_wd)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.args.c_epochs - 10], gamma=0.1)
            #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: CustomCosineAnnealing(epoch, first_task=task_id==0, total_epochs=self.args.c_epochs, initial_lr=1.0, min_lr=0.1))

        for epoch in range(self.args.c_epochs):

            train_acc_accum = 0
            train_loss_accum = 0


            self.model.train()

            if gen_train_loader is None:
                for i, batch in tqdm.tqdm(enumerate(train_loader)):


                    x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)

                    self.optimizer.zero_grad()
                    loss, acc = self.model.observe(x, y)


                    loss.backward()
                    # clip gradient norm
                    # if self.args.clip_grad_norm:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)


                    if self.args.c_sam:
                        self.optimizer.first_step(zero_grad=True)
                        loss, _ = self.model.observe(x, y)
                        loss.backward()
                        self.optimizer.second_step(zero_grad=True)

                    else:
                        self.optimizer.step()


                    train_acc_accum += acc.item()
                    train_loss_accum += loss.detach().item()

            
            else:
                
                # gen_train_loader = itertools.cycle(gen_train_loader)
                # gen_val_loader = itertools.cycle(gen_val_loader)
                for i, (batch, gen_batch) in tqdm.tqdm(enumerate(zip(train_loader, gen_train_loader))):
                    if self.args.c_ema:
                        ori_model = deepcopy(self.model)
                    x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
                    gen_x, gen_y = gen_batch['pixel_values'].to(self.device), gen_batch['cl_label'].to(self.device)

                    self.optimizer.zero_grad()
                    loss, acc = self.model.observe(x, y, gen_x, gen_y)
                    loss.backward()

                    # clip gradient norm
                    if self.args.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                        
                    if self.args.c_sam:
                        self.optimizer.first_step(zero_grad=True)
                        loss, _ = self.model.observe(x, y, gen_x, gen_y)
                        loss.backward()
                        self.optimizer.second_step(zero_grad=True)
                    
                    else:
                        self.optimizer.step()

                    if self.args.c_ema:
                        self.ema_update(self.model, ori_model)

                    train_acc_accum += acc.item()
                    train_loss_accum += loss.detach().item()

            train_acc_accum /= (i+1)
            train_acc_accum *= 100
            train_loss_accum /= (i+1)

            print(f"Epoch {epoch} - Train Loss: {train_loss_accum} - Train Acc: {train_acc_accum}% - lr: {self.scheduler.get_last_lr()[0]}")
            wandb.log({"Train Loss": train_loss_accum, "Train Acc": train_acc_accum})

            self.scheduler.step()

            self.model.eval()

            val_acc_accum = 0
            val_loss_accum = 0
            with torch.no_grad():

                if gen_val_loader is None:
                    for i, batch in enumerate(val_loader):
                        x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)

                        loss, acc = self.model.observe(x, y)

                        val_acc_accum += acc.item()
                        val_loss_accum += loss.item()

                else:
                    for i, (batch, gen_batch) in enumerate(zip(val_loader, gen_val_loader)):
                        x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
                        gen_x, gen_y = gen_batch['pixel_values'].to(self.device), gen_batch['cl_label'].to(self.device)

                        loss, acc = self.model.observe(x, y, gen_x, gen_y)

                        val_acc_accum += acc.item()
                        val_loss_accum += loss.item()

            val_acc_accum /= (i+1)
            val_acc_accum *= 100
            val_loss_accum /= (i+1)

            print(f"Epoch {epoch} - Val Loss: {val_loss_accum} - Val Acc: {val_acc_accum}%")
            wandb.log({"Val Loss": val_loss_accum, "Val Acc": val_acc_accum})

            accs = self.evaluate(test_loader_list, task_ids, self.device)
            mean_acc = np.mean(accs, axis=1)
            print_mean_accuracy(mean_acc, task_id)

            d2={'RESULT_step_class_mean_accs': mean_acc[0], 'RESULT_step_task_mean_accs': mean_acc[1],
                **{f'RESULT_step_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_step_task_acc_{i}': a for i, a in enumerate(accs[1])}}
            
            wandb.log(d2)

            print(d2)

        d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
            **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
            **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

        wandb.log(d2)
        print(d2)

        return accs
    
    # Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
    # All rights reserved.
    # This source code is licensed under the license found in the
    # LICENSE file in the root directory of this source tree.

    # Modified by ISAAC NING LEE


    def mask_classes(self, outputs: torch.Tensor, start_idx: int, end_idx: int) -> None:
        """
        Given the output tensor, the dataset at hand and the current task,
        masks the former by setting the responses for the other tasks at -inf.
        It is used to obtain the results for the task-il setting.
        :param outputs: the output tensor
        """
        outputs[:, 0:start_idx] = -float('inf')
        outputs[:, end_idx:] = -float('inf')


    def evaluate(self, test_loader_list: List, task_ids: dict, device) -> Tuple[list, list]:
        """
        Evaluates the accuracy of the model for each past task.
        :param model: the model to be evaluated
        :param dataset: the continual dataset at hand
        :return: a tuple of lists, containing the class-il
                and task-il accuracy for each task
        """
        self.model.eval()
        accs, accs_mask_classes = [], []

        start_id = 0
        end_id = 0
        for k, test_loader in enumerate(test_loader_list):

            end_id = len(task_ids[k]) + start_id

            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            for batch in test_loader:
                with torch.no_grad():
                    inputs, labels = batch['pixel_values'].to(device), batch['cl_label'].to(device)

                    outputs = self.model(inputs)

                    _, pred = torch.max(outputs, 1)
                    correct += torch.sum(pred == labels).item()
                    total += labels.shape[0]


                    self.mask_classes(outputs, start_id, end_id)
                    _, pred = torch.max(outputs, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

            accs.append(correct / total * 100)
            accs_mask_classes.append(correct_mask_classes / total * 100)

            start_id = end_id

        return accs, accs_mask_classes



# class DINOTrainer(Trainer):

#     def __init__(self, args, model: DINOBackbone, device):
#         super().__init__(args, model, device)

#     def train(self, task_id, task_ids: dict, train_loader, val_loader, test_loader_list, gen_train_loader=None, gen_val_loader=None):

#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.c_lr, weight_decay=self.args.c_wd)
#         self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30], gamma=0.1)

#         for epoch in range(self.args.c_epochs):

#             train_acc_accum = 0
#             train_loss_accum = 0


#             self.model.train()

#             if gen_train_loader is None:
#                 for i, batch in tqdm.tqdm(enumerate(train_loader)):
#                     x, y, prompts = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device), batch['text']

#                     self.optimizer.zero_grad()
#                     loss, acc = self.model.observe(x, y, prompts)


#                     loss.backward()
#                     self.optimizer.step()

#                     train_acc_accum += acc.item()
#                     train_loss_accum += loss.item()

            
#             else:

#                 for i, (batch, gen_batch) in tqdm.tqdm(enumerate(zip(train_loader, gen_train_loader))):
#                     x, y, prompts = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device), batch['text']
#                     gen_x, gen_y, gen_prompts = gen_batch['pixel_values'].to(self.device), gen_batch['cl_label'].to(self.device), gen_batch['text']

#                     self.optimizer.zero_grad()
#                     loss, acc = self.model.observe(x, y, prompts, gen_x, gen_y, gen_prompts)
#                     loss.backward()
#                     self.optimizer.step()

#                     train_acc_accum += acc.item()
#                     train_loss_accum += loss.item()

#             train_acc_accum /= len(train_loader)
#             train_acc_accum *= 100
#             train_loss_accum /= len(train_loader)

#             print(f"Epoch {epoch} - Train Loss: {train_loss_accum} - Train Acc: {train_acc_accum}%")
#             wandb.log({"Train Loss": train_loss_accum, "Train Acc": train_acc_accum})

#             self.scheduler.step()

#             self.model.eval()

#             val_acc_accum = 0
#             val_loss_accum = 0
#             with torch.no_grad():
#                 # print((self.model.fc.weight).norm(dim=0)[:self.model.class_range])
#                 # print(self.model.fc.bias)
#                 if gen_val_loader is None:
#                     for i, batch in enumerate(val_loader):
#                         x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)

#                         loss, acc = self.model.observe(x, y)

#                         val_acc_accum += acc.item()
#                         val_loss_accum += loss.item()

#                 else:
#                     for i, (batch, gen_batch) in enumerate(zip(val_loader, gen_val_loader)):
#                         x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
#                         gen_x, gen_y = gen_batch['pixel_values'].to(self.device), gen_batch['cl_label'].to(self.device)

#                         loss, acc = self.model.observe(x, y, gen_x, gen_y)

#                         val_acc_accum += acc.item()
#                         val_loss_accum += loss.item()

#             val_acc_accum /= len(val_loader)
#             val_acc_accum *= 100
#             val_loss_accum /= len(val_loader)

#             print(f"Epoch {epoch} - Val Loss: {val_loss_accum} - Val Acc: {val_acc_accum}%")
#             wandb.log({"Val Loss": val_loss_accum, "Val Acc": val_acc_accum})

#             accs = self.evaluate(test_loader_list, task_ids, self.device)
#             mean_acc = np.mean(accs, axis=1)
#             print_mean_accuracy(mean_acc, task_id)

#             d2={'RESULT_step_class_mean_accs': mean_acc[0], 'RESULT_step_task_mean_accs': mean_acc[1],
#                 **{f'RESULT_step_class_acc_{i}': a for i, a in enumerate(accs[0])},
#                 **{f'RESULT_step_task_acc_{i}': a for i, a in enumerate(accs[1])}}
            
#             wandb.log(d2)

#             print(d2)

#         d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
#             **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
#             **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

#         wandb.log(d2)
#         print(d2)


class AnnealTrainer(Trainer):

    def __init__(self, args, model: AnnealingBackbone, device):
        super().__init__(args, model, device)

        if self.args.use_c_l1:
            self.model.replace_RELU_with_SILU()
    

    def train(self, task_id, task_ids: dict, train_loader, val_loader, test_loader_list, gen_train_loader=None, gen_val_loader=None):


        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.c_lr, weight_decay=self.args.c_wd)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.args.c_epochs - 10], gamma=0.1)

        if task_id > 0:
            old_teacher = deepcopy(self.model)
            
        for epoch in range(self.args.c_epochs):

            train_acc_accum = 0
            train_loss_accum = 0


            self.model.train()




            for i, batch in tqdm.tqdm(enumerate(train_loader)):


                x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
                

                self.optimizer.zero_grad()
                loss, acc = self.model.observe(x, y)

                if self.args.use_c_l1:
                    l1_loss = self.model.l1_loss()
                    loss += self.args.c_wd * l1_loss


                loss.backward()
                # clip gradient norm
                # if self.args.clip_grad_norm:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)


                self.optimizer.step()


                train_acc_accum += acc.item()
                train_loss_accum += loss.detach().item()

            
            train_acc_accum /= (i+1)
            train_acc_accum *= 100
            train_loss_accum /= (i+1)

            print(f"Epoch {epoch} - Train Loss: {train_loss_accum} - Train Acc: {train_acc_accum}% - lr: {self.scheduler.get_last_lr()[0]}")
            wandb.log({"Train Loss": train_loss_accum, "Train Acc": train_acc_accum})

            self.scheduler.step()

            self.model.eval()

            val_acc_accum = 0
            val_loss_accum = 0
            with torch.no_grad():


                for i, batch in enumerate(val_loader):
                    x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)

                    loss, acc = self.model.observe(x, y)

                    val_acc_accum += acc.item()
                    val_loss_accum += loss.item()

            val_acc_accum /= len(val_loader)
            val_acc_accum *= 100
            val_loss_accum /= len(val_loader)

            print(f"Epoch {epoch} - Val Loss: {val_loss_accum} - Val Acc: {val_acc_accum}%")
            wandb.log({"Val Loss": val_loss_accum, "Val Acc": val_acc_accum})

            accs = self.evaluate(test_loader_list, task_ids, self.device)
            mean_acc = np.mean(accs, axis=1)
            print_mean_accuracy(mean_acc, task_id)

            d2={'RESULT_step_class_mean_accs': mean_acc[0], 'RESULT_step_task_mean_accs': mean_acc[1],
                **{f'RESULT_step_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_step_task_acc_{i}': a for i, a in enumerate(accs[1])}}
            
            wandb.log(d2)

            print(d2)

        if task_id == 0: 
            d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)
            print(d2)

        if task_id > 0:
            new_teacher = deepcopy(self.model)

            self.model.set_teachers(old_teacher, new_teacher)

            print("Starting the annealing phase ....")



            for epoch in range(self.args.c_anneal_epochs):

                train_acc_accum = 0
                train_loss_accum = 0


                self.model.train()
                # train_loader = itertools.cycle(train_loader)
                # val_loader = itertools.cycle(val_loader)

                for i, (batch, gen_batch) in tqdm.tqdm(enumerate(zip(train_loader, gen_train_loader))):

                    if self.args.c_ema:
                        ori_model = deepcopy(self.model)

                    x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
                    gen_x, gen_y = gen_batch['pixel_values'].to(self.device), gen_batch['cl_label'].to(self.device)

                    self.optimizer.zero_grad()
                    loss, acc = self.model.observe(x, y, gen_x, gen_y)
                    loss.backward()


                    self.optimizer.step()

                    if self.args.c_ema:
                        self.ema_update(self.model, ori_model, alpha=0.1)


                    train_acc_accum += acc.item()
                    train_loss_accum += loss.detach().item()

                
                train_acc_accum /= (i + 1)
                train_acc_accum *= 100
                train_loss_accum /= (i + 1)

                print(f"Epoch {epoch} - Train Loss: {train_loss_accum} - Train Acc: {train_acc_accum}% - lr: {self.scheduler.get_last_lr()[0]}")
                wandb.log({"Train Loss": train_loss_accum, "Train Acc": train_acc_accum})

                self.scheduler.step()

                self.model.eval()

                val_acc_accum = 0
                val_loss_accum = 0
                with torch.no_grad():
                    if gen_val_loader is None:
                        for i, batch in enumerate(val_loader):
                            x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)

                            loss, acc = self.model.observe(x, y)

                            val_acc_accum += acc.item()
                            val_loss_accum += loss.item()

                    else:
                        for i, (batch, gen_batch) in enumerate(zip(val_loader, gen_val_loader)):
                            x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
                            gen_x, gen_y = gen_batch['pixel_values'].to(self.device), gen_batch['cl_label'].to(self.device)

                            loss, acc = self.model.observe(x, y, gen_x, gen_y)

                            val_acc_accum += acc.item()
                            val_loss_accum += loss.item()

                val_acc_accum /= (i + 1)
                val_acc_accum *= 100
                val_loss_accum /= (i+1)

                print(f"Epoch {epoch} - Val Loss: {val_loss_accum} - Val Acc: {val_acc_accum}%")
                wandb.log({"Val Loss": val_loss_accum, "Val Acc": val_acc_accum})

                accs = self.evaluate(test_loader_list, task_ids, self.device)
                mean_acc = np.mean(accs, axis=1)
                print_mean_accuracy(mean_acc, task_id)

                d2={'RESULT_step_class_mean_accs': mean_acc[0], 'RESULT_step_task_mean_accs': mean_acc[1],
                    **{f'RESULT_step_class_acc_{i}': a for i, a in enumerate(accs[0])},
                    **{f'RESULT_step_task_acc_{i}': a for i, a in enumerate(accs[1])}}
                
                wandb.log(d2)

                print(d2)

            d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)
            print(d2)
        return accs


class AnnealTrainerPlus(Trainer):

    def __init__(self, args, model: AnnealingBackbone, device):
        super().__init__(args, model, device)

    def train(self, task_id, task_ids: dict, train_loader, val_loader, test_loader_list, gen_train_loader=None, gen_val_loader=None, aug_train_loader=None):


        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.c_lr, weight_decay=self.args.c_wd)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.args.c_epochs - 10], gamma=0.1)

        if task_id > 0:
            old_teacher = deepcopy(self.model)
            
        for epoch in range(self.args.c_epochs):

            train_acc_accum = 0
            train_loss_accum = 0


            self.model.train()

            for i, (batch, aug_batch) in tqdm.tqdm(enumerate(zip(train_loader, aug_train_loader))):


                x, y, x2 = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device), batch['pixel_values2'].to(self.device)
                aug_x, aug_y, aug_x2 = aug_batch['pixel_values'].to(self.device), aug_batch['cl_label'].to(self.device), aug_batch['pixel_values2'].to(self.device)

                x = torch.cat([x, aug_x, x2, aug_x2])
                y = torch.cat([y, aug_y, y, aug_y])

            
                self.optimizer.zero_grad()
                loss, acc = self.model.observe(x, y)

                if self.args.use_c_l1:
                    l1_loss = self.model.l1_loss()
                    loss += self.args.c_wd * l1_loss


                loss.backward()
                # clip gradient norm
                # if self.args.clip_grad_norm:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)


                self.optimizer.step()


                train_acc_accum += acc.item()
                train_loss_accum += loss.detach().item()

            
            train_acc_accum /= (i+1)
            train_acc_accum *= 100
            train_loss_accum /= (i+1)

            print(f"Epoch {epoch} - Train Loss: {train_loss_accum} - Train Acc: {train_acc_accum}% - lr: {self.scheduler.get_last_lr()[0]}")
            wandb.log({"Train Loss": train_loss_accum, "Train Acc": train_acc_accum})

            self.scheduler.step()

            self.model.eval()

            val_acc_accum = 0
            val_loss_accum = 0
            with torch.no_grad():


                for i, batch in enumerate(val_loader):
                    x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)

                    loss, acc = self.model.observe(x, y)

                    val_acc_accum += acc.item()
                    val_loss_accum += loss.item()

            val_acc_accum /= len(val_loader)
            val_acc_accum *= 100
            val_loss_accum /= len(val_loader)

            print(f"Epoch {epoch} - Val Loss: {val_loss_accum} - Val Acc: {val_acc_accum}%")
            wandb.log({"Val Loss": val_loss_accum, "Val Acc": val_acc_accum})

            accs = self.evaluate(test_loader_list, task_ids, self.device)
            mean_acc = np.mean(accs, axis=1)
            print_mean_accuracy(mean_acc, task_id)

            d2={'RESULT_step_class_mean_accs': mean_acc[0], 'RESULT_step_task_mean_accs': mean_acc[1],
                **{f'RESULT_step_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_step_task_acc_{i}': a for i, a in enumerate(accs[1])}}
            
            wandb.log(d2)

            print(d2)

        if task_id == 0: 
            d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)
            print(d2)

        if task_id > 0:
            new_teacher = deepcopy(self.model)

            self.model.set_teachers(old_teacher, new_teacher)

            print("Starting the annealing phase ....")

            #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.c_lr, weight_decay=self.args.c_wd, nesterov=True, momentum=0.9)

            for epoch in range(self.args.c_anneal_epochs):

                train_acc_accum = 0
                train_loss_accum = 0


                self.model.train()
                # train_loader = itertools.cycle(train_loader)
                # val_loader = itertools.cycle(val_loader)

                for i, (batch, gen_batch, aug_batch) in tqdm.tqdm(enumerate(zip(train_loader, gen_train_loader, aug_train_loader))):

                    if self.args.c_ema:
                        ori_model = deepcopy(self.model)

                    x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
                    gen_x, gen_y = gen_batch['pixel_values'].to(self.device), gen_batch['cl_label'].to(self.device)
                    aug_x, aug_y = aug_batch['pixel_values'].to(self.device), aug_batch['cl_label'].to(self.device)

                    x = torch.cat([x, aug_x])
                    y = torch.cat([y, aug_y])

                    self.optimizer.zero_grad()
                    loss, acc = self.model.observe(x, y, gen_x, gen_y)
                    loss.backward()


                    self.optimizer.step()

                    if self.args.c_ema:
                        self.ema_update(self.model, ori_model, alpha=0.1)


                    train_acc_accum += acc.item()
                    train_loss_accum += loss.detach().item()

                
                train_acc_accum /= (i + 1)
                train_acc_accum *= 100
                train_loss_accum /= (i + 1)

                print(f"Epoch {epoch} - Train Loss: {train_loss_accum} - Train Acc: {train_acc_accum}% - lr: {self.scheduler.get_last_lr()[0]}")
                wandb.log({"Train Loss": train_loss_accum, "Train Acc": train_acc_accum})

                self.scheduler.step()

                self.model.eval()

                val_acc_accum = 0
                val_loss_accum = 0
                with torch.no_grad():
                    if gen_val_loader is None:
                        for i, batch in enumerate(val_loader):
                            x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)

                            loss, acc = self.model.observe(x, y)

                            val_acc_accum += acc.item()
                            val_loss_accum += loss.item()

                    else:
                        for i, (batch, gen_batch) in enumerate(zip(val_loader, gen_val_loader)):
                            x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
                            gen_x, gen_y = gen_batch['pixel_values'].to(self.device), gen_batch['cl_label'].to(self.device)

                            loss, acc = self.model.observe(x, y, gen_x, gen_y)

                            val_acc_accum += acc.item()
                            val_loss_accum += loss.item()

                val_acc_accum /= (i + 1)
                val_acc_accum *= 100
                val_loss_accum /= (i+1)

                print(f"Epoch {epoch} - Val Loss: {val_loss_accum} - Val Acc: {val_acc_accum}%")
                wandb.log({"Val Loss": val_loss_accum, "Val Acc": val_acc_accum})

                accs = self.evaluate(test_loader_list, task_ids, self.device)
                mean_acc = np.mean(accs, axis=1)
                print_mean_accuracy(mean_acc, task_id)

                d2={'RESULT_step_class_mean_accs': mean_acc[0], 'RESULT_step_task_mean_accs': mean_acc[1],
                    **{f'RESULT_step_class_acc_{i}': a for i, a in enumerate(accs[0])},
                    **{f'RESULT_step_task_acc_{i}': a for i, a in enumerate(accs[1])}}
                
                wandb.log(d2)

                print(d2)

            d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)
            print(d2)
        return accs



    
class AugAnnealTrainer(Trainer):

    def __init__(self, args, model: AnnealingBackbone, device):
        super().__init__(args, model, device)

        if self.args.use_c_l1:
            self.model.replace_RELU_with_SILU()
    

    def train(self, task_id, task_ids: dict, train_loader, val_loader, test_loader_list, gen_train_loader=None, gen_val_loader=None, aug_train_loader=None):


        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.c_lr, weight_decay=self.args.c_wd)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.args.c_epochs - 10], gamma=0.1)

        if task_id > 0:
            old_teacher = deepcopy(self.model)
            
        for epoch in range(self.args.c_epochs):

            train_acc_accum = 0
            train_loss_accum = 0


            self.model.train()




            for i, (batch, aug_batch) in tqdm.tqdm(enumerate(zip(train_loader, aug_train_loader))):


                x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
                aug_x, aug_y = aug_batch['pixel_values'].to(self.device), aug_batch['cl_label'].to(self.device)

                x = torch.cat([x, aug_x])
                y = torch.cat([y, aug_y])

            
                self.optimizer.zero_grad()
                loss, acc = self.model.observe(x, y)

                if self.args.use_c_l1:
                    l1_loss = self.model.l1_loss()
                    loss += self.args.c_wd * l1_loss


                loss.backward()
                # clip gradient norm
                # if self.args.clip_grad_norm:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)


                self.optimizer.step()


                train_acc_accum += acc.item()
                train_loss_accum += loss.detach().item()

            
            train_acc_accum /= (i+1)
            train_acc_accum *= 100
            train_loss_accum /= (i+1)

            print(f"Epoch {epoch} - Train Loss: {train_loss_accum} - Train Acc: {train_acc_accum}% - lr: {self.scheduler.get_last_lr()[0]}")
            wandb.log({"Train Loss": train_loss_accum, "Train Acc": train_acc_accum})

            self.scheduler.step()

            self.model.eval()

            val_acc_accum = 0
            val_loss_accum = 0
            with torch.no_grad():


                for i, batch in enumerate(val_loader):
                    x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)

                    loss, acc = self.model.observe(x, y)

                    val_acc_accum += acc.item()
                    val_loss_accum += loss.item()

            val_acc_accum /= len(val_loader)
            val_acc_accum *= 100
            val_loss_accum /= len(val_loader)

            print(f"Epoch {epoch} - Val Loss: {val_loss_accum} - Val Acc: {val_acc_accum}%")
            wandb.log({"Val Loss": val_loss_accum, "Val Acc": val_acc_accum})

            accs = self.evaluate(test_loader_list, task_ids, self.device)
            mean_acc = np.mean(accs, axis=1)
            print_mean_accuracy(mean_acc, task_id)

            d2={'RESULT_step_class_mean_accs': mean_acc[0], 'RESULT_step_task_mean_accs': mean_acc[1],
                **{f'RESULT_step_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_step_task_acc_{i}': a for i, a in enumerate(accs[1])}}
            
            wandb.log(d2)

            print(d2)

        if task_id == 0: 
            d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)
            print(d2)

        if task_id > 0:
            new_teacher = deepcopy(self.model)

            self.model.set_teachers(old_teacher, new_teacher)

            print("Starting the annealing phase ....")

            #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.c_lr, weight_decay=self.args.c_wd, nesterov=True, momentum=0.9)

            for epoch in range(self.args.c_anneal_epochs):

                train_acc_accum = 0
                train_loss_accum = 0


                self.model.train()
                # train_loader = itertools.cycle(train_loader)
                # val_loader = itertools.cycle(val_loader)

                for i, (batch, gen_batch, aug_batch) in tqdm.tqdm(enumerate(zip(train_loader, gen_train_loader, aug_train_loader))):

                    if self.args.c_ema:
                        ori_model = deepcopy(self.model)

                    x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
                    gen_x, gen_y = gen_batch['pixel_values'].to(self.device), gen_batch['cl_label'].to(self.device)
                    aug_x, aug_y = aug_batch['pixel_values'].to(self.device), aug_batch['cl_label'].to(self.device)

                    x = torch.cat([x, aug_x])
                    y = torch.cat([y, aug_y])

                    self.optimizer.zero_grad()
                    loss, acc = self.model.observe(x, y, gen_x, gen_y)
                    loss.backward()


                    self.optimizer.step()

                    if self.args.c_ema:
                        self.ema_update(self.model, ori_model, alpha=0.1)


                    train_acc_accum += acc.item()
                    train_loss_accum += loss.detach().item()

                
                train_acc_accum /= (i + 1)
                train_acc_accum *= 100
                train_loss_accum /= (i + 1)

                print(f"Epoch {epoch} - Train Loss: {train_loss_accum} - Train Acc: {train_acc_accum}% - lr: {self.scheduler.get_last_lr()[0]}")
                wandb.log({"Train Loss": train_loss_accum, "Train Acc": train_acc_accum})

                self.scheduler.step()

                self.model.eval()

                val_acc_accum = 0
                val_loss_accum = 0
                with torch.no_grad():
                    if gen_val_loader is None:
                        for i, batch in enumerate(val_loader):
                            x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)

                            loss, acc = self.model.observe(x, y)

                            val_acc_accum += acc.item()
                            val_loss_accum += loss.item()

                    else:
                        for i, (batch, gen_batch) in enumerate(zip(val_loader, gen_val_loader)):
                            x, y = batch['pixel_values'].to(self.device), batch['cl_label'].to(self.device)
                            gen_x, gen_y = gen_batch['pixel_values'].to(self.device), gen_batch['cl_label'].to(self.device)

                            loss, acc = self.model.observe(x, y, gen_x, gen_y)

                            val_acc_accum += acc.item()
                            val_loss_accum += loss.item()

                val_acc_accum /= (i + 1)
                val_acc_accum *= 100
                val_loss_accum /= (i+1)

                print(f"Epoch {epoch} - Val Loss: {val_loss_accum} - Val Acc: {val_acc_accum}%")
                wandb.log({"Val Loss": val_loss_accum, "Val Acc": val_acc_accum})

                accs = self.evaluate(test_loader_list, task_ids, self.device)
                mean_acc = np.mean(accs, axis=1)
                print_mean_accuracy(mean_acc, task_id)

                d2={'RESULT_step_class_mean_accs': mean_acc[0], 'RESULT_step_task_mean_accs': mean_acc[1],
                    **{f'RESULT_step_class_acc_{i}': a for i, a in enumerate(accs[0])},
                    **{f'RESULT_step_task_acc_{i}': a for i, a in enumerate(accs[1])}}
                
                wandb.log(d2)

                print(d2)

            d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)
            print(d2)
        return accs
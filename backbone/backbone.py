import torch.nn as nn
import torchvision
import torch
import kornia
import kornia.augmentation as K
from cpsd.cpsd import CPSDPipeline
import math
import torch.nn.functional as F
from copy import deepcopy

class Backbone(nn.Module):

    def __init__(self, num_classes):
        super(Backbone, self).__init__()

        self.num_classes = num_classes

        self.net = torchvision.models.resnet18()
        
        self.fc = nn.Linear(in_features=self.net.fc.in_features, out_features=num_classes, bias=True)
        self.net.fc = nn.Identity()

        self.class_range = 0

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)[:, :self.class_range]
        return x

    def set_class_range(self, class_range):
        self.class_range = class_range

    def observe(self, x, y, gen_x = None, gen_y = None):

        if gen_x is None:

            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()

        else:
            x_cat = torch.cat([x, gen_x], dim=0)
            y_cat = torch.cat([y, gen_y], dim=0)

            y_hat = self(x_cat)
            loss = self.criterion(y_hat, y_cat)
            acc = (y_hat.argmax(dim=1) == y_cat).float().mean()

        return loss, acc
    
    def end_task(self):

        return None



class LUCIRBackbone(Backbone):

    def __init__(self, num_classes, zap_p=None):
        super(LUCIRBackbone, self).__init__(num_classes)

        self.weights = nn.ParameterList(
            []
        )
        self.sigma = nn.parameter.Parameter(torch.Tensor(1))

        self.in_features = self.fc.in_features
        self.old_net = None
        self.lambda_cos_sim = 0
        self.pc = 0
        self.nc = 0
        self.task_id = -1

        self.k_mr = 2
        self.mr_margin = 0.5
        self.lambda_mr = 1.0
        self.lambda_base = 5.0

        self.zap_p = zap_p

        self.reset_parameters()

    def add_task(self, num_new_classes, device):
        self.pc += self.nc
        self.task_id += 1
        self.nc = num_new_classes
        self.weights.append(nn.Parameter(torch.Tensor(num_new_classes, self.in_features)).to(device))
        self.reset_parameters()
        
    def reset_parameters(self):
        for i in range(len(self.weights)):
            stdv = 1. / math.sqrt(self.weights[i].size(1))
            self.weights[i].data.uniform_(-stdv, stdv)
            self.weights[i].requires_grad = False

        self.sigma.data.fill_(1)
        self.old_net = deepcopy(self.net)
        self.old_fc = deepcopy(self.fc)

        self.lambda_cos_sim = math.sqrt(len(self.weights)) * float(self.lambda_base)

    # def zap(self):

    #     if self.zap_p:
    #         # check if random number less than zap_p
    #         current_id = self.task_id - 1
    #         if torch.rand(1).item() < self.zap_p:
    #             stdv = 1. / math.sqrt(self.weights[current_id].size(1))
    #             self.weights[current_id].data.uniform_(-stdv, stdv)
    #             self.weights[current_id].requires_grad = False
    
    def forward(self, x):

        x = self.net(x)
        return self.noscale_forward(x) * self.sigma
    
    def reset_weight(self, i):
        stdv = 1. / math.sqrt(self.weights[i].size(1))
        self.weights[i].data.uniform_(-stdv, stdv)
        self.weights[i].requires_grad = True
        self.weights[i - 1].requires_grad = False

    def noscale_forward(self, x):


        out = None

        x = F.normalize(x, p=2, dim=1).reshape(len(x), -1)

        for t in range(len(self.weights)):
            o = F.linear(x, F.normalize(self.weights[t], p=2, dim=1))
            if out is None:
                out = o
            else:
                out = torch.cat((out, o), dim=1)

        return out


    def observe(self, x, y, gen_x=None, gen_y=None):
        
        # self.zap()
        if self.task_id == 0:

            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()

        else:
            if gen_x is None:
                x_cat = x
                y_cat = y
            else:
                x_cat = torch.cat([x, gen_x], dim=0)
                y_cat = torch.cat([y, gen_y], dim=0)


            z_hat = self.net(x_cat)
            y_hat = self.noscale_forward(z_hat) * self.sigma
            loss = self.criterion(y_hat, y_cat)

            z_hat = z_hat.reshape(z_hat.size(0), -1)

            with torch.no_grad():

                logits = self.old_net(x_cat)
                logits = logits.reshape(logits.size(0), -1)

            loss2 = F.cosine_embedding_loss(
                z_hat, logits.detach(), torch.ones(z_hat.shape[0]).to(z_hat.device)) * self.lambda_cos_sim

            loss3 = self.lucir_batch_hard_triplet_loss(
                y_cat, y_hat, self.k_mr, self.mr_margin, self.pc) * self.lambda_mr
                
            loss += loss2 + loss3

            
            # fool_loss = self.fool_loss(z_hat, y, gen_y) * self.lambda_mr 
            # loss += fool_loss

            acc = (y_hat.argmax(dim=1) == y_cat).float().mean()

        return loss, acc

    # def fool_loss(self, z_hat, y, gen_y):

    #     len_x = len(y)

    #     z_hat_x = z_hat[:len_x, :]
    #     z_hat_gen_x = z_hat[len_x:, :]

    #     # Compute pairwise cosine similarity
    #     z_hat_x_norm = F.normalize(z_hat_x, p=2, dim=1)
    #     z_hat_gen_x_norm = F.normalize(z_hat_gen_x, p=2, dim=1)
    #     cosine_sim = torch.mm(z_hat_x_norm, z_hat_gen_x_norm.t())

    #     # Create labels for contrastive loss
    #     positive_pairs = (y.unsqueeze(1) == gen_y.unsqueeze(0)).float()

    #     # Compute contrastive loss
    #     margin = 1.0  # Margin for contrastive loss
    #     fool_loss = (1 - positive_pairs) * F.relu(margin - cosine_sim) + positive_pairs * (1 - cosine_sim)
    #     fool_loss = fool_loss.mean()

    #     return fool_loss

    def lucir_batch_hard_triplet_loss(self, labels, embeddings, k, margin, num_old_classes):
        """
        LUCIR triplet loss.
        """
        gt_index = torch.zeros(embeddings.size()).to(embeddings.device)
        gt_index = gt_index.scatter(1, labels.reshape(-1, 1).long(), 1).ge(0.5)
        gt_scores = embeddings.masked_select(gt_index)
        # get top-K scores on novel classes
        max_novel_scores = embeddings[:, num_old_classes:].topk(k, dim=1)[0]
        # the index of hard samples, i.e., samples of old classes
        hard_index = labels.lt(num_old_classes)
        hard_num = torch.nonzero(hard_index).size(0)
        if hard_num > 0:
            gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, k)
            max_novel_scores = max_novel_scores[hard_index]
            assert (gt_scores.size() == max_novel_scores.size())
            assert (gt_scores.size(0) == hard_num)
            target = torch.ones(hard_num * k, 1).to(embeddings.device)
            loss = nn.MarginRankingLoss(margin=margin)(gt_scores.view(-1, 1),
                                                    max_novel_scores.view(-1, 1), target)
        else:
            loss = torch.zeros(1).to(embeddings.device)[0]

        return loss

class DINOBackbone(Backbone):
    
    def __init__(self, num_classes, pipeline:CPSDPipeline, c_resolution):
        super(DINOBackbone, self).__init__(num_classes)

        self.pipeline = pipeline
        self.crop1 = K.RandomResizedCrop(size=(c_resolution, c_resolution), scale=(0.4, 1.0), keepdim=True, resample=kornia.constants.Resample.BICUBIC)
        self.crop2 = K.RandomResizedCrop(size=(c_resolution, c_resolution), scale=(0.75, 1.0), keepdim=True, resample=kornia.constants.Resample.BICUBIC)
        self.normalize = K.Normalize(mean=0.5, std=0.5)
        self.jitter = K.ColorJitter(0.05, 0.05, 0.05, 0.02)
        self.blur = K.RandomBoxBlur(kernel_size=(3, 3), p=0.5)
        self.mse_loss = nn.MSELoss()

    def augment(self, x, prompts):
        
        with torch.no_grad():
            # random crop
            x1 = self.crop1(x)
            x2 = self.crop1(x)

            # boomerang aug
            x1 = self.jitter(self.blur(x1))
            x2 = self.jitter(self.blur(x2))

            # crop again
            x1 = self.crop2(x1)
            x2 = self.crop2(x2)

            # normalize
            x1 = self.normalize(x1).detach()
            x2 = self.normalize(x2).detach()

        return x1, x2

    def forward(self, x, return_z=False):
        z = self.net(x)
        y = self.fc(z)[:, :self.class_range]

        if return_z:

            return y, z
        else:
            return y

    def observe(self, x, y, prompt=None, gen_x=None, gen_y=None, gen_prompt=None):

        if self.training:

            x1, x2 = self.augment(x, prompt)

            if gen_x is None:

                y_hat_1, z_hat_1 = self(x1, return_z=True)
                y_hat_2, z_hat_2 = self(x2, return_z=True)

                y_hat_cat = torch.cat([y_hat_1, y_hat_2], dim=0)
                y_cat = torch.cat([y, y], dim=0)

                

            else:
                gen_x1, gen_x2 = self.augment(gen_x, gen_prompt)

                x_cat_1 = torch.cat([x1, gen_x1], dim=0)
                x_cat_2 = torch.cat([x2, gen_x2], dim=0)

                y_cat = torch.cat([y,y, gen_y, gen_y], dim=0)

                y_hat_1, z_hat_1 = self(x_cat_1, return_z=True)
                y_hat_2, z_hat_2 = self(x_cat_2, return_z=True)

                y_hat_cat = torch.cat([y_hat_1, y_hat_2], dim=0)

            loss = self.criterion(y_hat_cat, y_cat) + 1e-2 * self.mse_loss(z_hat_1, z_hat_2)
            acc = (y_hat_cat.argmax(dim=1) == y_cat).float().mean()
            return loss, acc
        else:
            return super().observe(x, y, gen_x, gen_y)


class AnnealingBackbone(Backbone):

    def __init__(self, num_classes, anti_discrim=False, init_option='old'):
        super(AnnealingBackbone, self).__init__(num_classes)

        self.old_teacher = None
        self.new_teacher = None
        self.mse_loss = nn.MSELoss()

        self.mem_alpha = 0.5
        self.anti_discrim = anti_discrim
        self.init_option = init_option

        self.task_id = -1

    def forward(self, x, return_z=False):
        z = self.net(x)
        x = self.fc(z)[:, :self.class_range]

        if return_z:
            return x, z
        else:
            return x

    def set_class_range(self, class_range):
        self.class_range = class_range
        self.task_id += 1


    def set_teachers(self, old_teacher, new_teacher):

        self.old_teacher = old_teacher
        self.new_teacher = new_teacher

        # # init self.net and self.fc as old_teacher, cannot use deepcopy as optimizer will break

        if self.init_option == 'old':
            with torch.no_grad():
                for param, old_param in zip(self.net.parameters(), self.old_teacher.net.parameters()):
                    param.data = old_param.data.clone()

                for param, old_param in zip(self.fc.parameters(), self.old_teacher.fc.parameters()):
                    param.data = old_param.data.clone()

        elif self.init_option == 'new':
            with torch.no_grad():
                for param, new_param in zip(self.net.parameters(), self.new_teacher.net.parameters()):
                    param.data = new_param.data.clone()

                for param, new_param in zip(self.fc.parameters(), self.new_teacher.fc.parameters()):
                    param.data = new_param.data.clone()

        elif self.init_option == 'mean':
            with torch.no_grad():
                for param, old_param, new_param in zip(self.net.parameters(), self.old_teacher.net.parameters(), self.new_teacher.net.parameters()):
                    param.data = (old_param.data + new_param.data) / 2

                for param, old_param, new_param in zip(self.fc.parameters(), self.old_teacher.fc.parameters(), self.new_teacher.fc.parameters()):
                    param.data = (old_param.data + new_param.data) / 2
                    
        elif self.init_option == 'norm_mean':

            self.mem_alpha = 1 - 1 / (self.task_id + 1)
            with torch.no_grad():
                for param, old_param, new_param in zip(self.net.parameters(), self.old_teacher.net.parameters(), self.new_teacher.net.parameters()):
                    param.data = F.normalize((old_param.data), p=2, dim=1) * self.mem_alpha + F.normalize((new_param.data), p=2, dim=1) * (1 - self.mem_alpha)

                for param, old_param, new_param in zip(self.fc.parameters(), self.old_teacher.fc.parameters(), self.new_teacher.fc.parameters()):
                    param.data = F.normalize((old_param.data ), p=2, dim=1) * self.mem_alpha + F.normalize((new_param.data), p=2, dim=1) * (1 - self.mem_alpha)

        elif self.init_option == 'random':
            self.net = torchvision.models.resnet18()
            self.fc = nn.Linear(in_features=self.net.fc.in_features, out_features=self.num_classes, bias=True)
            self.net.fc = nn.Identity()


    def end_task(self):

        self.old_teacher = None
        self.new_teacher = None

        return super().end_task()         

    def observe(self, x, y, gen_x = None, gen_y = None):

        if gen_x is None:

            x_cat = x
            y_cat = y

        else:
            x_cat = torch.cat([x, gen_x], dim=0)
            y_cat = torch.cat([y, gen_y], dim=0)

        mse_loss = 0.0
        if self.old_teacher is not None and self.training and gen_x is not None:
            
            with torch.no_grad():
                _ , z_old = self.old_teacher(x_cat, return_z=True)
                _, z_new = self.new_teacher(x_cat, return_z=True)
            
            y_hat, z_student = self(x_cat, return_z=True)
            mse_loss = self.mem_alpha * self.mse_loss(z_student, z_old) + (1 - self.mem_alpha) * self.mse_loss(z_student, z_new)

            if self.anti_discrim:
                gen_start_idx = len(x)
                ce_loss = self.criterion(y_hat[gen_start_idx:], y_cat[gen_start_idx:]) # prevent discriminatory effect
            else:
                ce_loss = self.criterion(y_hat, y_cat)

            loss = ce_loss + mse_loss
            acc = (y_hat.argmax(dim=1) == y_cat).float().mean()

        else:
            y_hat = self(x_cat)
            loss = self.criterion(y_hat, y_cat)
            acc = (y_hat.argmax(dim=1) == y_cat).float().mean()

        return loss, acc


        


        
        


            

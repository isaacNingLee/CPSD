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



class LUCIRBackbone(Backbone):

    def __init__(self, num_classes):
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

        self.k_mr = 2
        self.mr_margin = 0.5
        self.lambda_mr = 1.0
        self.lambda_base = 5.0

        self.reset_parameters()

    def add_task(self, num_new_classes, device):
        self.pc += self.nc
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

        if gen_x is None:

            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()

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

            acc = (y_hat.argmax(dim=1) == y_cat).float().mean()

        return loss, acc

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





        


        
        


            
import torch.nn as nn
import torchvision
import torch
import kornia
import kornia.augmentation as K
from cpsd.cpsd import CPSDPipeline
import math
import torch.nn.functional as F
from copy import deepcopy
import math

class Backbone(nn.Module):

    def __init__(self, args, num_classes, device):
        super(Backbone, self).__init__()

        self.num_classes = num_classes

        self.net = torchvision.models.resnet18()
        
        self.fc = nn.Linear(in_features=self.net.fc.in_features, out_features=num_classes, bias=True)
        self.net.fc = nn.Identity()

        self.device = device

        self.args = args

        if args.c_resolution < 100:
            self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.net.maxpool = nn.Identity()

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

    def __init__(self, args, num_classes, zap_p=None):
        super(LUCIRBackbone, self).__init__(args, num_classes)

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






class DINOAugment(nn.Module):

    def __init__(self, c_resolution):
        super(DINOAugment, self).__init__()

        self.crop1 = K.RandomResizedCrop(size=(c_resolution, c_resolution), scale=(0.4, 1.0), keepdim=True, resample=kornia.constants.Resample.BICUBIC)
        self.crop2 = K.RandomResizedCrop(size=(c_resolution, c_resolution), scale=(0.75, 1.0), keepdim=True, resample=kornia.constants.Resample.BICUBIC)
        self.normalize = K.Normalize(mean=0.5, std=0.5)
        self.jitter = K.ColorJitter(0.05, 0.05, 0.05, 0.02)
        self.blur = K.RandomBoxBlur(kernel_size=(3, 3), p=0.5)

    def forward(self, x):

        x = self.crop1(x)
        x = self.jitter(self.blur(x))
        x = self.crop2(x)
        x = self.normalize(x)

        return x
    
class AnnealingBackbone(Backbone):

    def __init__(self, args, num_classes, anti_discrim=False, init_option='old', plus=False, device='cuda'):
        super(AnnealingBackbone, self).__init__(args, num_classes, device=device)

        self.old_teacher = None
        self.new_teacher = None
        self.mse_loss = nn.MSELoss()
        #self.dino_augment = DINOAugment(224)

        # self.contrast_head = nn.Linear(in_features=self.fc.in_features, out_features=128, bias=True)

        self.mem_alpha = 0.5
        self.anti_discrim = anti_discrim
        self.init_option = init_option

        self.task_id = -1
        self.plus = plus

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

        if self.init_option == 'mean':
            with torch.no_grad():
                for param, old_param, new_param in zip(self.net.parameters(), self.old_teacher.net.parameters(), self.new_teacher.net.parameters()):
                    param.data = (old_param.data + new_param.data) / 2

                for param, old_param, new_param in zip(self.fc.parameters(), self.old_teacher.fc.parameters(), self.new_teacher.fc.parameters()):
                    param.data = (old_param.data + new_param.data) / 2


        elif self.init_option == 'random':
            self.net = torchvision.models.resnet18()
            self.fc = nn.Linear(in_features=self.net.fc.in_features, out_features=self.num_classes, bias=True)
            self.net.fc = nn.Identity()
            self.to(self.device)

    def end_task(self):

        self.old_teacher = None
        self.new_teacher = None

        return super().end_task()


    # def class_con_loss2(self, z, z_a, y):

    #     # Compute pairwise cosine similarity
    #     z_norm = F.normalize(z, p=2, dim=1)
    #     z_a_norm = F.normalize(z_a, p=2, dim=1)
    #     cosine_sim = torch.mm(z_norm, z_a_norm.t())

    #     # prepapre labels where col is y and row is gen_y, for each row, the label is 1 if the row is the same as the col
    #     positive_pairs = (y.unsqueeze(1) == y.unsqueeze(0)).float()


    #     # Compute contrastive loss
    #     loss = torch.nn.functional.binary_cross_entropy_with_logits(cosine_sim, positive_pairs)

    #     return loss

    # def class_con_loss(self, z, y_cat):
            
    #     # randomly split z into z1 and z2
    #     idx = torch.randperm(z.size(0))
    #     z1 = z[idx[:len(y_cat)//2]]
    #     z2 = z[idx[len(y_cat)//2:]]

    #     # Compute pairwise cosine similarity
    #     z_norm_1 = F.normalize(z1, p=2, dim=1)
    #     z_norm_2 = F.normalize(z2, p=2, dim=1)
    #     cosine_sim = torch.mm(z_norm_1, z_norm_2.t())

    #     # prepapre labels where col is y and row is gen_y, for each row, the label is 1 if the row is the same as the col
    #     positive_pairs = (y_cat[idx[:len(y_cat)//2]].unsqueeze(1) == y_cat[idx[len(y_cat)//2:]].unsqueeze(0)).float()

    #     # Compute contrastive loss
    #     loss = torch.nn.functional.binary_cross_entropy_with_logits(cosine_sim, positive_pairs)

    #     return loss
    
    # def info_nce_loss(self, features):

    #     if features.shape[0] <= 1:
    #         return torch.tensor(0.0).to(self.device)
        
    #     features = self.contrast_head(features)

    #     labels = torch.cat([torch.arange(features.shape[0] // 2) for i in range(2)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(self.device)

    #     features = F.normalize(features, dim=1)

    #     similarity_matrix = torch.matmul(features, features.T)
    #     # assert similarity_matrix.shape == (
    #     #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    #     # assert similarity_matrix.shape == labels.shape

    #     # discard the main diagonal from both: labels and similarities matrix
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     # assert similarity_matrix.shape == labels.shape

    #     # select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    #     # select only the negatives the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    #     logits = torch.cat([positives, negatives], dim=1)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

    #     logits = logits / 0.07
    #     loss = self.criterion(logits, labels)

    #     return loss
    
    # def replace_RELU_with_SILU(self):
    #     # for sparsity to work
            
    #     for name, module in self.net.named_children():
    #         if isinstance(module, nn.ReLU):
    #             setattr(self.net, name, nn.SiLU())


    # def l1_loss(self):
    #     loss = 0
    #     for param in self.net.parameters():
    #         loss += torch.norm(param, 1)

    #     for param in self.fc.parameters():
    #         loss += torch.norm(param, 1)

    #     return loss



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
                _ , z_new = self.new_teacher(x_cat, return_z=True)

            
            y_hat, z_student = self(x_cat, return_z=True)
            mse_loss = self.mem_alpha * self.mse_loss(z_student, z_old) + (1 - self.mem_alpha) * self.mse_loss(z_student, z_new)


            if self.anti_discrim:
                gen_start_idx = len(x)
                ce_loss = self.criterion(y_hat[gen_start_idx:], y_cat[gen_start_idx:]) + 0.1 * (1 - self.mem_alpha) * self.criterion(y_hat[:gen_start_idx], y_cat[:gen_start_idx])

            else:
                ce_loss = self.criterion(y_hat, y_cat)


            loss = ce_loss + mse_loss
            acc = (y_hat.argmax(dim=1) == y_cat).float().mean()

        else:
            y_hat, z_hat = self(x_cat, return_z=True)



            loss = self.criterion(y_hat, y_cat)

            acc = (y_hat.argmax(dim=1) == y_cat).float().mean()

        return loss, acc
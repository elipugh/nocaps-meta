
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    updown.models.updown_captioner import UpDownCaptioner
from    copy import deepcopy



class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, config, vocabulary):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = config.DATA.UPDATE_LR
        self.meta_lr = config.DATA.META_LR
        self.k_spt = config.DATA.K_SPT
        self.k_qry = config.DATA.K_QRY
        self.update_step = config.DATA.UPDATE_STEP
        self.update_step_test = config.DATA.UPDATE_STEP_TEST
        self.vocabulary=vocabulary
        self.device = torch.device("cuda:0")
        self.net = UpDownCaptioner.from_config(config, vocabulary=vocabulary).to(self.device)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.sd = self.net.state_dict()



    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, batch):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz, maxlen, d_emb]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz, maxlen, d_emb]
        :return:
        """
        task_num = len(batch)

        losses_q = [0 for _ in range(self.update_step + 1)]

        for it in range(task_num):
            x,y = batch[it]["image_features"], batch[it]["caption_tokens"]
            x_spt, x_qry = x[:self.k_spt], x[self.k_spt:]
            y_spt, y_qry = y[:self.k_spt], y[self.k_spt:]
            x_spt, x_qry = x_spt.to(self.device), x_qry.to(self.device)
            y_spt, y_qry = y_spt.to(self.device), y_qry.to(self.device)
            # 1. run the i-th task and compute loss for k=0
            self.net.train()
            self.meta_optim.zero_grad()

            output_dict = self.net(x_spt, y_spt)

            loss = output_dict["loss"].mean()

            for param in self.net.parameters():
                param.requires_grad = True

            params = []
            for k,v in self.net.state_dict().items():
                if v.requires_grad or True:
                    params += [v]
                    params[-1].requires_grad = True

            grad = torch.autograd.grad(loss, params, allow_unused=True)
            print(grad)
            params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, params)))

            sd2 = deepcopy(self.sd)
            i = 0
            for k,v in self.net.state_dict().items():
                if v.requires_grad or True:
                    sd2[k] = params[i]
                    i += 1

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                self.net.load_state_dict(self.sd)
                output_dict_q = self.net(x_qry, y_qry)
                loss_q = output_dict_q["loss"].mean()
                losses_q[0] += loss_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                self.net.load_state_dict(sd2)
                output_dict_q = self.net(x_qry, y_qry)
                loss_q = output_dict_q["loss"].mean()
                losses_q[1] += loss_q

            self.net.train()

            for k in range(1, self.update_step):
                for param in self.net.parameters():
                    param.requires_grad = True
                # 1. run the i-th task and compute loss for k=1~K-1
                self.net.load_state_dict(sd2)
                self.net.train()
                output_dict = self.net(torch.unsqueeze(x_spt[it],0), torch.unsqueeze(y_spt[it],0))
                loss = output_dict["loss"].mean()
                # 2. compute grad on theta_pi
                params = []
                for _,v in self.net.state_dict().items():
                    if v.requires_grad or True:
                        params += [v]
                        params[-1].requires_grad = True
                print(len(params))
                grad = torch.autograd.grad(loss, params)
                # 3. theta_pi = theta_pi - train_lr * grad
                params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, params)))

                i = 0
                for key,v in self.net.state_dict().items():
                    if v.requires_grad or True:
                        sd2[key] = params[i]
                        i += 1
                self.net.load_state_dict(sd2)
                output_dict_q = self.net(torch.unsqueeze(x_qry[it],0), torch.unsqueeze(y_qry[it],0))
                loss_q = output_dict_q["loss"].mean()
                losses_q[k + 1] += loss_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        #   print(torch.norm(p).item())
        self.meta_optim.step()

        return loss_q

    def finetuning(self, batch):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        x,y = batch[0]["image_features"], batch[0]["caption_tokens"]
        x_spt, x_qry = x[:self.k_spt], x[self.k_spt:]
        y_spt, y_qry = y[:self.k_spt], y[self.k_spt:]
        x_spt, x_qry = x_spt.to(self.device), x_qry.to(self.device)
        y_spt, y_qry = y_spt.to(self.device), y_qry.to(self.device)

        sd2 = deepcopy(self.sd)
        losses_q = [0 for _ in range(self.update_step + 1)]
        self.net.load_state_dict(sd2)
        self.net.train()
        self.meta_optim.zero_grad()


        output_dict = self.net(x_spt,y_spt)
        loss = output_dict["loss"].mean()

        params = []
        for k,v in self.sd.items():
            if v.requires_grad:
                params += [v]
        grad = torch.autograd.grad(loss, params, retain_graph=True)
        params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, params)))
        i = 0
        for k,v in self.sd.items():
            if v.requires_grad:
                sd2[k] = params[i]
                i += 1

        with torch.no_grad():
            self.net.load_state_dict(self.sd)
            output_dict_q = self.net(x_spt,y_spt)
            loss_q = output_dict_q["loss"].mean()
            losses_q[0] += loss_q

        with torch.no_grad():
            self.net.load_state_dict(sd2)
            output_dict_q = self.net(x_spt,y_spt)
            loss_q = output_dict_q["loss"].mean()
            losses_q[1] += loss_q

        sd3 = deepcopy(sd2)
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            output_dict = self.net(x_spt, y_spt)
            loss = output_dict["loss"].mean()
            params = []
            for _,v in self.sd.items():
                if v.requires_grad:
                    params += [v]

            grad = torch.autograd.grad(loss, params, retain_graph=True)
            params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, params)))
            i = 0
            for key,v in self.sd.items():
                if v.requires_grad:
                    sd2[key] = params[i]
                    i += 1

            self.net.load_state_dict(sd2)
            output_dict = self.net(x_spt, y_spt)
            loss_q = output_dict["loss"].mean()
            losses_q[k+1] += loss_q

            # for key,v in sd3.items():
            #     print((v-sd2[key]).mean(), end="\r")
            # print()

        return losses_q



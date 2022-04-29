import os
import math
import torch
import numpy as np
from tools.sgd import SGD
import torch.autograd.variable as Variable
from tools.logistic import logistic_F
from tools.softmax import softmax_F
from tensorboardX import SummaryWriter
from tools.lib import *
from torch import nn


def train_model(taskid_path, args, net, train_dataloader, val_dataloader, density, dataset_length):
    def backward(grad_output):
        x, weight, bias, mask_weight, padding, label, mask, Iter, density, parameter_filter, parameter_mag, parameter_sqrtvar, parameter_strength, parameter_sliceMag = self.saved_tensors

        input = x.mul(torch.max(mask, torch.zeros(mask.size()).cuda()))
        if self.needs_input_grad[0]:
            x_grad = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, padding=int(padding.item()))
        if self.needs_input_grad[1]:
            weight_grad = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, padding=int(padding.item()))
        if bias is not None and self.needs_input_grad[2]:
            bias_grad = grad_output.sum(0).sum((1, 2))

        depth = x.size()[1]
        h = x.size()[2]
        w = x.size()[3]
        depthList = (parameter_filter > 0).nonzero()[:, 0].reshape(-1, 1)
        labelNum = label.size()[1]
        Div_list = []

        if (labelNum == 1):
            theClass = label
            posList = (theClass == 1).nonzero()[:, 0].reshape(-1, 1)
            Div = Divclass(depthList, posList)
            Div_list.append(Div)
        else:
            (theClass, indextmp) = torch.max(label, dim=1)
            theClass = theClass.unsqueeze(2)
            if (parameter_sliceMag.size()[0] == torch.Tensor([]).size()[0]):
                posList = (theClass == 1).nonzero()[:, 0].reshape(-1, 1)
                Div = Divclass(depthList, posList)
                Div_list.append(Div)
            else:
                sliceM = parameter_sliceMag
                for i in range(labelNum):
                    Div = Divclass(depthList=torch.Tensor([]), posList=torch.Tensor([]))
                    Div_list.append(Div)
                (val, index) = torch.max(sliceM[depthList, :].squeeze(1), dim=1)
                for lab in range(labelNum):
                    (Div_list[lab].depthList, indextmp) = torch.sort(depthList[index == lab], dim=0)
                    Div_list[lab].posList = (label[:, lab, :, :] == 1).nonzero()[:, 0].reshape(-1, 1)

        imgNum = label.size()[0]
        alpha = 169/170
        x_grad = x_grad.mul(torch.max(mask, torch.zeros(mask.size()).cuda()))

        if ((torch.sum(parameter_filter == 1)) > 0):
            parameter_strength = torch.mean(torch.mean(x.mul(mask), 2), 2).transpose(1, 0).cuda()
            mask_tmp = (torch.from_numpy(copy.deepcopy(mask.cpu().detach().numpy()[::-1, ::-1, :, :]))).cuda()
            alpha_logZ_pos = (torch.log(torch.mean(torch.exp(torch.mean(torch.mean(x.mul(mask_tmp), 2), 2).div(alpha)), 0)) * alpha).reshape(depth, 1)
            alpha_logZ_neg = (torch.log(torch.mean(torch.exp(torch.mean(torch.mean(-x, 2), 2).div(alpha)), 0)) * alpha).reshape(depth, 1)

            # restrict
            #alpha_logZ_pos[alpha_logZ_pos > 10000.] = torch.tensor(10000.).cuda()
            #alpha_logZ_pos[alpha_logZ_pos < -10000.] = torch.tensor(-10000.).cuda()

            #alpha_logZ_neg[alpha_logZ_neg > 10000.] = torch.tensor(10000.).cuda()
            #alpha_logZ_neg[alpha_logZ_neg < -10000.] = torch.tensor(-10000.).cuda()

            alpha_logZ_pos[torch.isinf(alpha_logZ_pos)] = torch.max(alpha_logZ_pos[torch.isinf(alpha_logZ_pos) == 0])
            alpha_logZ_neg[torch.isinf(alpha_logZ_neg)] = torch.max(alpha_logZ_neg[torch.isinf(alpha_logZ_neg) == 0])

        for lab in range(len(Div_list)):
            if (labelNum == 1):
                w_pos = 0.5/169 * max(-1, 1-(4/13))
                w_neg = -0.5/169
            else:
                if (labelNum > 10):
                    w_pos = 0.5 / (1 / labelNum)
                    w_neg = 0.5 / (1 - 1 / labelNum)
                else:
                    w_pos = 0.5 / density[lab]
                    w_neg = 0.5 / (1 - density[lab])

            mag = torch.ones([depth, imgNum]).div(1 / Iter).div(parameter_mag).cuda()
            dList = Div_list[lab].depthList
            dList = dList[(parameter_filter[dList] == 1).squeeze(1)].reshape(-1, 1)
            if (dList.size()[0] != torch.Tensor([]).size()[0]):
                List = Div_list[lab].posList.cuda()
                if (List.size()[0] != torch.Tensor([]).size()[0]):
                    strength = torch.exp((parameter_strength[:, List].squeeze(2))[dList, :].squeeze(1).div(alpha)).mul((parameter_strength[:, List].squeeze(2))[dList, :].squeeze(1) - alpha_logZ_pos[dList].squeeze(1).repeat(1, List.size()[0]) + alpha)
                    # restrict
                    #strength[strength > 10000.] = torch.tensor(10000.).cuda()
                    #strength[strength < -10000.] = torch.tensor(-10000.).cuda()
                    strength[torch.isinf(strength)] = torch.max(strength[torch.isinf(strength) == 0])
                    strength[torch.isnan(strength)] = 0
                    strength = (strength.div((torch.mean(strength, 1).reshape(-1, 1).repeat(1, List.size()[0])).mul((mag[:, List].squeeze(2))[dList, :].squeeze(1)))).transpose(0, 1).reshape(List.size()[0],dList.size()[0], 1, 1)
                    strength[torch.isnan(strength)] = 0
                    # restrict
                    #strength[strength > 10000.] = torch.tensor(10000.).cuda()
                    #strength[strength < -10000.] = torch.tensor(-10000.).cuda()
                    strength[torch.isinf(strength)] = torch.max(strength[torch.isinf(strength) == 0])
                    index_dList = dList.repeat(List.size()[0], 1)
                    index_List = List.reshape(-1, 1).repeat(1, dList.size()[0]).reshape(List.size()[0] * dList.size()[0], 1)
                    x_grad[index_List, index_dList, :, :] = ((x_grad[List, :, :, :].squeeze(1))[:, dList, :, :].squeeze(2) - (mask[List, :, :,:].squeeze(1))[:,dList,:,:].squeeze(2).mul(strength.repeat(1, 1, h, w) * (0.00001 * w_pos))).reshape(List.size()[0] * dList.size()[0],1, h, w)

                list_neg = (label != 1).nonzero()[:, 0].reshape(-1, 1)
                if (list_neg.size()[0] != torch.Tensor([]).size()[0]):
                    strength = torch.mean((torch.mean((x[list_neg, :, :, :].squeeze(1))[:, dList, :, :].squeeze(2), 2).unsqueeze(2)),3).unsqueeze(2).transpose(0, 1).reshape(dList.size()[0], list_neg.size()[0])
                    strength = torch.exp(-strength.div(alpha)).mul(-strength - alpha_logZ_neg[dList].squeeze(2).repeat(1, list_neg.size()[0]) + alpha)
                    # restrict
                    #strength[strength > 10000.] = torch.tensor(10000.).cuda()
                    #strength[strength < -10000.] = torch.tensor(-10000.).cuda()
                    strength[torch.isinf(strength)] = torch.max(strength[torch.isinf(strength) == 0])
                    strength[torch.isnan(strength)] = 0
                    strength = (strength.div((torch.mean(strength, 1).reshape(-1, 1).repeat(1, list_neg.size()[0])).mul((mag[:, list_neg].squeeze(2))[dList, :].squeeze(1)))).transpose(0, 1).reshape(list_neg.size()[0], dList.size()[0], 1, 1)
                    strength[torch.isnan(strength)] = 0
                    # restrict
                    #strength[strength > 10000.] = torch.tensor(10000.).cuda()
                    #strength[strength < -10000.] = torch.tensor(-10000.).cuda()
                    strength[torch.isinf(strength)] = torch.max(strength[torch.isinf(strength) == 0])
                    index_dList = dList.repeat(list_neg.size()[0], 1)
                    index_list_neg = list_neg.reshape(-1, 1).repeat(1, dList.size()[0]).reshape(list_neg.size()[0] * dList.size()[0], 1)
                    x_grad[index_list_neg, index_dList, :, :] = ((x_grad[list_neg, :, :, :].squeeze(1))[:, dList, :, :].squeeze(2) + (strength.reshape(list_neg.size()[0], dList.size()[0], 1, 1).repeat(1, 1, h, w)) * (0.00001 * w_neg)).reshape(list_neg.size()[0] * dList.size()[0], 1, h, w)

        beta = 4.0
        mask_weight_grad = torch.zeros(depth, 1).cuda()
        parameter_sqrtvar = parameter_sqrtvar.transpose(0, 1)

        for lab in range(len(Div_list)):
            dList = Div_list[lab].depthList.cuda()
            List = Div_list[lab].posList
            if ((dList.size()[0] != torch.Tensor([]).size()[0]) and (List.size()[0] != torch.Tensor([]).size()[0])):
                tmp = ((torch.sum((parameter_strength[:, List].squeeze(2))[dList, :].squeeze(1).mul((parameter_sqrtvar[:, List].squeeze(2))[dList, :].squeeze(1)), 1)).
                       div(torch.sum((parameter_strength[:, List].squeeze(2))[dList, :].squeeze(1), 1))).reshape(-1, 1)
                tmptmp = beta / tmp
                tmp = torch.max(torch.min(tmptmp, torch.ones(tmptmp.size()).cuda() * 3),torch.ones(tmptmp.size()).cuda() * 1.5)
                tmp = (tmp - mask_weight[dList].squeeze(2)) * (-10000)
                mask_weight_grad[dList] = tmp.unsqueeze(2)

        return x_grad, weight_grad, bias_grad, mask_weight_grad, None, None, None, None, None, None, None, None, None, None, None

    log_path = os.path.join(taskid_path,"log")
    make_dir(log_path)
    writer = SummaryWriter(log_path)

    torch.cuda.set_device(args.gpu_id)
    net = net.cuda()

    max_acc = 0
    max_epoch = 0
    judge = 0


    for epoch in range(args.epochnum):
        paras = dict(net.named_parameters())
        paras_new = []
        for k, v in paras.items():
            if 'mask' in k:
                if 'bias' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 2, 'weight_decay': args.weightdecay * 0}]
                if 'mask_weight' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 0.05, 'weight_decay': args.weightdecay * 0}]
                if '.weight' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 1, 'weight_decay': args.weightdecay * 1}]
            if 'line' in k:
                if 'bias' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 2, 'weight_decay': args.weightdecay * 0}]
                if 'weight' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 1, 'weight_decay': args.weightdecay * 1}]
            if 'conv' in k:
                if 'bias' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 1, 'weight_decay': args.weightdecay * 1}]
                if 'weight' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 1, 'weight_decay': args.weightdecay * 1}]
        optimizer = SGD(paras_new, lr=args.lr[epoch], momentum=args.momentum, weight_decay=args.weightdecay)

        # train
        net.train()
        train_loss = []
        train_acc = []
        print('Train: ' + "\n" + 'epoch:{}'.format(epoch + 1))
        for index, (image, label) in enumerate(train_dataloader):
            batch_size = image.shape[0]

            image = Variable(image)
            image = image.cuda()
            label = label.cuda()

            out = net(image, label, torch.Tensor([epoch + 1]), density)

            if args.model == "resnet_18" or args.model == "resnet_50" or args.model == "densenet_121":
                out = torch.unsqueeze(out,2)
                out = torch.unsqueeze(out, 3)
            label = Variable(label)
            if args.losstype == 'logistic':
                loss = logistic_F.apply(out, label)
                train_loss.append(loss.cpu().clone().data.numpy())
                train_correct = label.mul(out)
                train_correct = torch.max(train_correct, torch.zeros(train_correct.size()).cuda())
                train_correct = torch.sum((train_correct > 0))
                train_acc.append(train_correct.cpu().data.numpy())
            if args.losstype == 'softmax':
                loss = softmax_F.apply(out, label)
                train_loss.append(loss.cpu().clone().data.numpy())
                (tmp, out) = torch.sort(out, dim=1, descending=True)
                (tmp, label) = torch.max(label, dim=1)
                label = label.unsqueeze(2)
                error = ~(out == label)
                train_correct = args.batchsize - torch.sum(error[:, 0, 0, 0])
                train_acc.append(train_correct.cpu().data.numpy())
            optimizer.zero_grad()
            loss=backward(loss)
            optimizer.step()

            print('batch:{}/{}'.format(index + 1, len(train_dataloader)) + " " +
                  'loss:{:.6f}'.format(loss / batch_size) + " " +
                  'acc:{:.6f}'.format(train_correct.cpu().data.numpy()/(batch_size*args.label_num)))

            length = dataset_length['train'] if index + 1 == len(train_dataloader) else args.batchsize * (index + 1)
            if (index + 1) % 10:
                writer.add_scalar('Train/Loss', sum(train_loss)/ length, epoch)
                writer.add_scalar('Train/acc', sum(train_acc)/ (length*args.label_num), epoch)


        # eval

        net.eval()
        with torch.no_grad():
            eval_loss = []
            eval_acc = []
            for index, (image, label) in enumerate(val_dataloader):
                print('Val: ' + "\n" + 'epoch:{}'.format(epoch + 1))
                batch_size = image.shape[0]
                image = Variable(image)
                image = image.cuda()
                label = label.cuda()

                out = net(image, label, torch.Tensor([epoch + 1]), density)
                if args.model == "resnet_18" or args.model == "resnet_50" or args.model == "densenet_121":
                    out = torch.unsqueeze(out, 2)
                    out = torch.unsqueeze(out, 3)
                label = Variable(label)
                if args.losstype == 'logistic':
                    loss = logistic_F.apply(out, label)
                    eval_loss.append(loss.cpu().data.numpy())
                    eval_correct = label.mul(out)
                    eval_correct = torch.max(eval_correct, torch.zeros(eval_correct.size()).cuda())
                    eval_correct = torch.sum((eval_correct > 0))
                    eval_acc.append(eval_correct.cpu().data.numpy())
                if args.losstype == 'softmax':
                    loss = softmax_F.apply(out, label)
                    eval_loss.append(loss.cpu().data.numpy())
                    (tmp, out) = torch.sort(out, dim=1, descending=True)
                    (tmp, label) = torch.max(label, dim=1)
                    label = label.unsqueeze(2)
                    error = ~(out == label)
                    eval_correct = args.batchsize - torch.sum(error[:, 0, 0, 0])
                    eval_acc.append(eval_correct.cpu().data.numpy())
                length = dataset_length['val'] if index + 1 == len(val_dataloader) else args.batchsize * (index + 1)
                print('batch:{}/{}'.format(index + 1, len(val_dataloader)) + " " +
                      'loss:{:.6f}'.format(loss/batch_size) + " " +
                      'acc:{:.6f}'.format(eval_correct.cpu().data.numpy()/(batch_size*args.label_num)))
            print("max_acc:"+str(max_acc))

            if sum(eval_acc)/(length*args.label_num)>max_acc:
                judge=1
                max_acc=sum(eval_acc)/(length*args.label_num)
                print("rightnow max_acc:"+str(max_acc))
                max_epoch=epoch

            writer.add_scalar('Eval/Loss', sum(eval_loss)/ length, epoch)
            writer.add_scalar('Eval/acc', sum(eval_acc)/ (length*args.label_num), epoch)
        if judge==1 or (epoch+1)%50==0:
            # save
            torch.save(net, taskid_path + '/net-' + str(epoch + 1) + '.pkl')
            #torch.save(net.state_dict(), taskid_path + '/net-params-' + str(epoch + 1) + '.pkl')
            judge=0

    return max_acc,max_epoch+1

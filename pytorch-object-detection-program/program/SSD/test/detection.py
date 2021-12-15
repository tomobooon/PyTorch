#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from myfunctions import decode, nms

class Detect(Function):
    def forward(self, output, num_classes, 
                top_k=200, variance=[0.1,0.2], 
                conf_thresh=0.01, nms_thresh=0.45):    
        loc_data, conf_data, prior_data = output[0], output[1], output[2]
        # conf_data �͊e�N���X�̐M���x�A[ bs, 8732, num_classes ]
        # �M���x�̕����� softmax �Ŋm���ɒ���
        softmax = nn.Softmax(dim=-1)
        conf_data = softmax(conf_data)       
        # num  �� batch �̑傫�� 
        num = loc_data.size(0)  
        # �o�͂̔z��������A���g�͍��� 0 �A[ bs, 21, 200, 5 ]
        output = torch.zeros(num, num_classes, top_k, 5)
        # conf_data [bs, 8732, num_classes] �� 
        # [bs,num_classes,8732]  �ɕό`���� conf_preds �Ɩ��t����
        conf_preds = conf_data.transpose(2, 1)
        # Decode predictions into bboxes.
        for i in range(num):  # �o�b�`���̊e�f�[�^�̏���
            # loc_data �� DBox ���� BBox ���쐬
            decoded_boxes = decode(loc_data[i], prior_data, variance)
            # conf_preds �� conf_scores �Ƀn�[�h�R�s�[
            conf_scores = conf_preds[i].clone()
            for cl in range(1, num_classes): # �e�N���X�̏���
                # conf_scores �ŐM���x�� conf_thresh �ȏ�� index �����߂�
                c_mask = conf_scores[cl].gt(conf_thresh)
                # conf_thresh �ȏ�̐M���x�̏W�������
                scores = conf_scores[cl][c_mask]
                # ���̏W���̗v�f���� 0�A�܂�conf_thresh �ȏ�͂Ȃ�
                # ����ȍ~�̏����͂Ȃ��ŁA���̃N���X��
                if scores.size(0) == 0:
                    continue
                # c_mask �� decoded_boxes �ɓK�p�ł���悤�ɃT�C�Y�ύX
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask �� decoded_boxes �ɓK�p�A�A�A�P�����ɂȂ�
                # view(-1, 4) �ŃT�C�Y��߂�
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # boxes �ɑ΂��� nms ��K�p�A
                # ids �� nms ��ʉ߂��� BBox �� index 
                # count �� nms ��ʉ߂��� BBox �̐�
                ids, count = nms(boxes, scores, nms_thresh, top_k)
                # ��L�̌��ʂ� output �Ɋi�[
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        return output


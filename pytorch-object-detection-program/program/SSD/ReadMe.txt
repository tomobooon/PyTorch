-----------------------------------------------
�����葁�� SSD �̊w�K���邢�͐��_���������@
-----------------------------------------------

�{���̂Q�͂Ŏ����� SSD �̊w�K�v���O����(myssd0.py)�Ŏ��ۂɃ��f�����w�K
�����Ă��A�Ȃ��Ȃ����x�͏オ��Ȃ��Ǝv���܂��B���ۂɎg����w�K�v���O��
���͂R�͂� myssd3.py �ł��B

���̃f�B���N�g���i./SSD�j�ł͎����葁�� SSD �̊w�K���邢�͐��_������
�̂ɕK�v�ȃt�@�C���������W�߂܂����B�����ɋL�����菇���o�邱�ƂŁASSD
�̊w�K���邢�͐��_���s���܂��B

------------------------------------------------
�� �w�K�����������ꍇ

(0) �f�B���N�g�� train �Ɉړ�

����Ō��݂̃f�B���N�g���i�J�����g�f�B���N�g���j��./SSD/train �ɂȂ��
���B

(1) �w�K�f�[�^�̃_�E�����[�h�ƓW�J

�ȉ��̃t�@�C�����_�E�����[�h���A�J�����g�f�B���N�g���i./SSD/train�j��
���œW�J���ĉ������B

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

(2) ank.pkl �̍쐬

�J�����g�f�B���N�g���i./SSD/train�j�̉��ňȉ������s���ĉ������B

python ann2list.py

�J�����g�f�B���N�g���i./SSD/train�j�� ans.pkl �̃t�@�C�����쐬����܂��B

(3) vgg �̏������f�[�^�̃_�E�����[�h

�ȉ��̃t�@�C�����_�E�����[�h���A�J�����g�f�B���N�g���i./SSD/train�j��
���ɒu���ĉ������B

https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

(4) �w�K

�ȏ�Ŋw�K���s���܂��B

python myssd3.py

�v���O�������̈ȉ��̐ݒ�͎��g�̊��ƖړI�ɍ��킹�āA�K���ɕύX���ĉ�
�����B

batch_size = 30  ## �o�b�`�T�C�Y�AGPU �̃����������Ȃ��ꍇ�A�����Ə���������
epoch_num = 15   ## �w�K�̃G�|�b�N��

------------------------------------------------
�� ���_�����������ꍇ

(0) �f�B���N�g�� test �Ɉړ�

����Ō��݂̃f�B���N�g���i�J�����g�f�B���N�g���j�� ./SSD/test �ɂȂ��
���B

(1) ���f���̃_�E�����[�h

���̊������f�����_�E�����[�h���āA�J�����g�f�B���N�g���i./SSD/test�j��
���ɒu���ĉ������B

https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth

����͎��g�Ŋw�K�������f���i�Ⴆ�� mymodel.bin�j�ł��\���܂���B

(2) ���o�Ώۂ̉摜�t�@�C���̏���

���o�ΏۂƂȂ�摜�t�@�C���i�����ł� mydog.png�j���J�����g�f�B���N�g��
�i./SSD/test�j�̉��ɒu���ĉ������B

(3) ���_

�ȏ�Ō��o�Ώۂɑ΂��ĕ��̌��o���s���܂��B

python mytest.py mydog.png ssd300_mAP_77.43_v2.pth

���f��(ssd300_mAP_77.43_v2.pth)�̕����́A���g���\�z�������f��
(mymodel.bin) �⑼�� SSD �̃��f���ł��\���܂���B






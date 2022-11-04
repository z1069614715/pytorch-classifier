import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['SoftTarget', 'MGD', 'SP', 'AT']

class SoftTarget(nn.Module):
    def __init__(self, T=4):
        super(SoftTarget, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.T = T
    
    def forward(self, student_pred, teacher_pred):
        student_pred_logsoftmax = torch.log_softmax(student_pred / self.T, dim=1)
        teacher_pred_softmax = torch.softmax(teacher_pred / self.T, dim=1)
        kd_loss = self.kl_loss(student_pred_logsoftmax, teacher_pred_softmax) * self.T * self.T
        return kd_loss

    def __str__(self):
        return 'SoftTarget'

class MGD(nn.Module):
    """PyTorch version of `Masked Generative Distillation`
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Defaults to 0.15
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.00007,
                 lambda_mgd=0.15,
                 ):
        super(MGD, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
    
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))


    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T)*self.alpha_mgd
            
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,C,1,1)).to(device)
        mat = torch.where(mat < self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss
    
    def __str__(self):
        return 'MGD'

class SP(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self):
        super(SP, self).__init__()

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return F.normalize(torch.matmul(z, torch.t(z)), 1)

    def forward(self, fm_s, fm_t):
        g_t = self.matmul_and_normalize(fm_t)
        g_s = self.matmul_and_normalize(fm_s)

        sp_loss = torch.norm(g_t - g_s) ** 2
        sp_loss = sp_loss.sum()
        return sp_loss / (fm_s.size(0) ** 2)
    
    def __str__(self):
        return 'SP'

class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self):
		super(AT, self).__init__()

	def forward(self, fm_s, fm_t):
		fm_s_att = self.attention_map(fm_s)
		fm_t_att = self.attention_map(fm_t)

		return (fm_s_att - fm_t_att).pow(2).mean()

	def attention_map(self, x):
		return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

	def __str__(self):
		return 'AT'
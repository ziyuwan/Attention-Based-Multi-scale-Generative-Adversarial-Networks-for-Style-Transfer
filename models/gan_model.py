import torch

from .base_model import BaseModel
from .dis_model import DisModel


class GANModel():
    def __init__(self, gen: BaseModel, dis: DisModel):
        super().__init__()
        self.gen = gen
        self.dis = dis
        self.use_dis = False

    def generate(self, data):
        self.gen.set_input(data)
        self.gen.forward()
        if self.use_dis:
            self.dis.set_input({'s_true': None, 's_fake': self.gen.cs, 's_super': data['s_super']})
            loss = self.dis.forward_generate()
            self.gen.loss_dis = loss
        self.gen.optimize_parameters()

    def discriminate(self, data):
        self.gen.set_input(data)
        with torch.no_grad():
            self.gen.forward()
        self.dis.set_input({'s_true': self.gen.s, 's_fake': self.gen.cs, 's_super': data['s_super']})
        self.dis.forward()
        self.dis.optimize_parameters()

from .train_options import TrainOptions


class GANOptions(TrainOptions):
    def initialize(self, parser):
        parser = super(GANOptions, self).initialize(parser)
        parser.add_argument('--lr_dis', type=float, default=1e-4, help='initial learning rate for dis')
        return parser

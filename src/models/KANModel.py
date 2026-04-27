from torch import nn
import torch.nn.functional as F

from ..KANLayer import KANLinear

class KANNet(nn.Module):
    def __init__(self, input_size=12):
        super().__init__()

        self.kan1 = KANLinear(
            input_size,
            20,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-3, 3],
        )

        '''self.kan2 = KANLinear(
            30,
            30,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-3, 3],
        )

        self.kan3 = KANLinear(
            30,
            30,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-3, 3],
        )

        self.kan4 = KANLinear(
            30,
            30,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-3, 3],
        )

        self.kan5 = KANLinear(
            30,
            30,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-3, 3],
        )

        self.kan6 = KANLinear(
            30,
            30,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-3, 3],
        )'''

        self.kan7 = KANLinear(
            20,
            20,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-3, 3],
        )

        self.kan8 = KANLinear(
            20,
            10,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-3, 3],
        )

        self.kan9 = KANLinear(
            10,
            3,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-3, 3],
        )

    def forward(self, x):
        x = self.kan1(x)
        '''x = self.kan2(x)
        x = self.kan3(x)
        x = self.kan4(x)
        x = self.kan5(x)
        x = self.kan6(x)'''
        x = self.kan7(x)
        x = self.kan8(x)
        x = self.kan9(x)
        x = F.log_softmax(x, dim=1)

        return x

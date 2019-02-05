import torch
import torch.nn.functional as F


def chabanne_2(x):
    # x = x/2
    return 0.1992 + 0.5002 * x + 0.1997 * x ** 2


def chabanne_3(x):
    return 0.1995 + 0.5002 * x + 0.1994 * x ** 2 - 0.0164 * x ** 3


def chabanne_4(x):
    return 0.1500 + 0.5012 * x + 0.2981 * x ** 2 - 0.0004 * x ** 3 - 0.0388 * x ** 4


def chabanne_5(x):
    return 0.1488 + 0.4993 * x + 0.3007 * x ** 2 + 0.0003 * x ** 3 - 0.0168 * x ** 5


def chabanne_6(x):
    return 0.1249 + 0.5000 * x + 0.3729 * x ** 2 - 0.0410 * x ** 4 + 0.0016 * x ** 6


def d3_v1_pol(x):
    return 0.7 * x ** 3 + 0.8 * x ** 2 + 0.2 * x


def d3_v2_pol(x):
    return -0.4 * x ** 3 + 0.5 * x ** 2 + 0.9 * x


def softplus_integral(x):
    return -0.0005 * x ** 4 + 0.0000 * x ** 3 + 0.0815 * x ** 2 + 0.5000 * x + 0

softplus = torch.nn.Softplus()


def custom_softplus(x):
    return x - softplus(x)


def hesam_sigmoid_integral(x):
    return -(x * (225234375 * x ** 3 + 443 * x ** 2 - 843750000000000 * x - 937500000000000000)) / 1875000000000000000


def bounded_step_activation(x):
    return torch.abs(x) * 0.5


def rectified_polynomial(x):
    # Rectifying all negative values first
    x = x.clamp(min=0)
    return d3_v2_pol(x)


def swish(x, beta=1):
    return x * F.sigmoid(beta * x)


def periodic_cos(x):
    return torch.cos(x) - x


def periodic_cos_mod(x):
    return torch.cos(0.2 * x) - (0.2 * x)


def softplus_polynomial(x):
    return -8.043291176102489*10**-14*x**9 -5.409176004846577*10**-11*x**8 +1.464006789445581*10**-10*x**7 +1.2094736421337893*10**-7*x**6 -8.68650047151514*10**-8*x**5 -9.849521136327391*10**-5*x**4 +1.8543655255840298*10**-5*x**3 +0.045459999581864446*x**2 +0.4989722694638288*x +1.1980867140213445

import numpy as np
import matplotlib.pyplot as plt
import struct
from math import sqrt


def as_u32(f):
    s = struct.pack('=f', f)
    u = struct.unpack('=I', s)[0]
    # print(f, '=>',  u)
    return u

def as_f32(u):
    s = struct.pack('=I', u)
    f = struct.unpack('=f', s)[0]
    # print(u, '=>', f)
    return f


class Approx:
    def __init__(self, name, c1, c2, c3, xmax):
        self.name = name
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.xmax = xmax

    def eval(self, x):
        u = as_u32(x)
        u = self.c1 - (u >> 1)
        f = as_f32(u)
        yapprox = self.c2 * f * (self.c3 - x*f*f)
        ytrue = 1.0/sqrt(x)
        return abs(yapprox - ytrue) / ytrue
        # y.u = 0x5f1ffff9 - (y.u >> 1); 0.703952253f * y.f * (2.389244556f - x * y.f * y.f)

    def plot(self):
        x = np.linspace(1.0, 4.0, num=512)
        # c3 = np.linspace(1, 6, num=512)
        # c2 = np.linspace(0.2, 1.0, num=512)
        x = np.append(x, self.xmax)
        x.sort()
        y = [self.eval(xx) for xx in x]
        plt.plot(x, y, '-', label=self.name)
        plt.annotate('{}'.format(self.eval(self.xmax)),
            (self.xmax, self.eval(self.xmax)), color=plt.gca().lines[-1].get_color())

def main():
   quake_orig = Approx('quake_orig', 0x5f3759df, 0.5, 3.0, 1)
   rr_lsq = Approx('rr_lsq', 0x5f1ad0a1,0.755897697, 2.27828001, 1)
   rr_best = Approx('rr_best', 0x5f1ffff9,0.703952253, 2.38924456, 1.9316406)
   my_best1 = Approx('my_best1', 0x5f1ffcee, 0.703950703, 2.389243603, 1)
   my_best2 = Approx('my_best2', 0x5f5f9f17, 0.250249714, 4.761075497, 1)
   # my_best3 = Approx('my_best3', 0x5f2031e1, 0.698473573, 2.401543140)

   plt.figure(1)
   # quake_orig.plot()
   # rr_lsq.plot()
   rr_best.plot()
   my_best1.plot()
   my_best2.plot()
   plt.grid()
   plt.legend(loc='lower right', ncol=1, frameon=False)
   plt.show()

if __name__ == '__main__':
    main()

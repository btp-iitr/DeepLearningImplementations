from xlrd import open_workbook
from math import log2
import numpy as np
from collections import deque

wb = open_workbook('data.xlsx')  # file containing the data
r_score = []           # stores the overall score of an individual on real images(1D)
f_score = []           # stores the overall score of an individual on generated images(1D)
r_image = []           # stores the scores given by different individuals to each real image(2D) 
f_image = []           # stores the scores given by different individuals to each generated image(2D)
ref = [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]  # 1=real;0=generated
real = deque()
fake = deque()
for s in wb.sheets():
    for i in range(1, s.nrows):
        rs = 0
        fs = 0
        real.clear()
        fake.clear()
        for j in range(1, s.ncols):
            p = s.cell(i, j).value
#            if type(p) != float:
#                print(i)
#                print(j)
            if p <= 0.05:
                p = 0.05        # lower and upper limit on the probability to avoid a loss of -INF
            elif p >= 0.95:
                p = 0.95
            if ref[j-1] == 0:
                fs += log2(2*p)
                fake.appendleft(p)
            else:
                rs += log2(2*p)
                real.appendleft(p)
        r_image.append(list(real))
        f_image.append(list(fake))
        r_score.append(rs)
        f_score.append(fs)

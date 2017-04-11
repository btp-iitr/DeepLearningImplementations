from xlrd import open_workbook
from math import log2
wb = open_workbook('data.xlsx')   #file containing the data
r_score = []
f_score = []
ref = [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
for s in wb.sheets():
    for i in range(1, s.nrows):
        rs = 0
        fs = 0
        for j in range(1, s.ncols):
            p = s.cell(i, j).value
            if p <= 0.05:
                p = 0.05        #lower and upper limit on the probability to avoid a loss of -INF
            elif p >= 0.95:
                p = 0.95
            if ref[j-1] == 0:
                fs += log2(2*p)
            else:
                rs += log2(2*p)
        r_score.append(rs)
        f_score.append(fs)

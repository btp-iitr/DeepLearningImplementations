from xlrd import open_workbook
from math import log2
wb = open_workbook('data.xlsx')   #file containing the data
r_score = []
f_score = []
for s in wb.sheets():
    rs = 0
    fs = 0
    for i in range(1, s.nrows):
        p = s.cell(i, 1).value
        if p <= 0.05:
            p = 0.05        #lower and upper limit on the probability to avoid a loss of -INF
        elif p >= 0.95:
            p = 0.95
        if s.cell(i, 0).value == 0:
            fs += log2(2*p)
        else:
            rs += log2(2*p)
    r_score.append(rs)
    f_score.append(fs)

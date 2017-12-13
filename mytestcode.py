# -*- coding: utf-8 -*-


from decimal import *
from helper import *
from fractions import Fraction



# 1 矩阵运算

A = [[2,1,2],
     [3,2,1],
     [1,2,1],
     [1,1,1]]
B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]
b = [[1],[2],[3],[4]]


def shape(M):
    r = len(M)
    c = len(M[0])
    return r,c

def matxRound(M, decPts=4):
    for p in M:
        for index in range(len(M[0])):
            p[index] = round(p[index], decPts)

def transpose(M):
    trans_M = zip(*M)
    return trans_M

def transpose(M):
    c_trans, r_trans = shape(M)
    trans_M = [] # 会造成ascii编码错误
    for i in range(r_trans):
        new_row = [0]*c_trans
        for j in range(c_trans):
            new_row[j] = M[j][i]
        new_row = tuple(new_row)
        trans_M.append(new_row)
    return trans_M

def matxMultiply(A, B):
    r_A = len(A)
    c_A = len(A[0])
    r_B = len(B)
    c_B = len(B[0])
    multi_C = [[0]*c_B for i in range(r_A)]
    if c_A != r_B:
        raise ValueError("A can't mutiply with B")
    else:
        for i in range(r_A):
            for j in range(c_B):
                for k in range(c_A):
                    multi_C[i][j] += A[i][k]*B[k][j]
    return multi_C

def matxMultiply(A, B):
    result = [[0]*len(B[0]) for i in range(len(A))]
    if len(A[0]) != len(B):
        raise ValueError("A can't mutiply with B")
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k]*B[k][j]
    return result

def matxMultiply(A, B):
    _, na = shape(A)
    mb, _ = shape(B)
    if na!=mb:
        return None
    Bt = transpose(B)
    result = [[sum((a*b) for a,b in zip(row,col)) for col in Bt] for row in A]
    return result
# print matxMultiply(A, B)

def augmentMatrix(A, b):
    result = [[0]*(len(A[0])+1) for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])+1):
            if j < len(A[0]):
                result[i][j] = A[i][j]
            else:
                result[i][j] = b[i][0]
    return result

#print augmentMatrix(A, b)


# 2 Gaussign Jordan 消元法

def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]

def scaleRow(M, r, scale):
    if scale != 0:
        for i in range(len(M[r])):
            M[r][i] = M[r][i]*scale
    else:
        raise ValueError("scale is 0")

# TODO r1 <--- r1 + r2*scale
def addScaledRow(M, r1, r2, scale):
    if scale != 0:
        for i in range(len(M[r1])):
            M[r1][i] += M[r2][i]*scale
    else:
        raise ValueError("scale is 0")


A = generateMatrix(4,5,singular=False)
b = np.ones(shape=(4,1),dtype=int) # it doesn't matter
Ab = augmentMatrix(A.tolist(),b.tolist()) # 请确保你的增广矩阵已经写好了
#printInMatrixFormat(Ab,padding=3,truncating=0)
#print Ab



def trans_to_fraction(M):
    for i in range(len(M)):
        for j in range(len(M[0])):
            M[i][j] = Fraction(M[i][j])

def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    if len(A) != len(b):    #如果 A，b 高度不同，返回None
        return None
    M = augmentMatrix(A.tolist(),b.tolist())
    #print M
    for i in range(len(M[0])-1):    #取第i列
        #print 'i:',i
        value_list = [0]*(len(M)-i)
        for j in range(i,len(M)):
            value = M[j][i]
            value_list[j-i] = abs(value)
        #print 'value_list:',value_list    #对角线及对角线以下值绝对值的列表
        max_value = max(value_list)    #取第i列对角线及以下数值绝对值的最大值
        #print 'max_value:',max_value
        if max_value == epsilon:    #如果最大值为0，则A为奇异矩阵，返回None
            return None
        else:
            for r in range(len(value_list)):
                if value_list[r] == max_value:    #取绝对值最大值所在行
                    #print '最大值所在行:',M[r+i]
                    #print '第i行:',M[i]
                    swapRows(M, i, r+i)    #绝对值最大值所在行与i行交换
                    #print '交换后的第i行:',M[i]
                    scaleRow(M, i, 1.0/M[i][i])     #1.0/M[i][i] 第i行对角线值变为1
                    #print '对角线值为1的第i行',M[i]
                    for l in range(len(M)):    #第i列除对角线值外的元素变为0
                        if l != i:
                            #print '第l行:',l
                            #print "第l行",M[l]
                            addScaledRow(M, l, i, -M[l][i])    #第i列第l个值变为0
                            #print '第i列第l个值为0的l行:',M[l]
                        else:
                            continue
                    #print M    #每一列循环后的M
                    break    #退出r的循环
    '''for i in range(len(M))[::-1]:    #从最后一行开始循环
        #print 'i',i
        for j in range(i):    #从第一行开始循环至第i行的上一行结束
            #print 'j',j
            addScaledRow(M, j, i, -M[j][i])    #第i列的第j个值变为0
            #print '第i列的第j个值变为0的第j行',M[j]'''
    print M
    x = [0]*len(b)
    for i in range(len(M)):
        x[i] = round(Decimal(M[i][-1]),decPts)
        
    return x
                    

# print gj_Solve(A, b, decPts=4, epsilon = 1.0e-16)




print 3**3










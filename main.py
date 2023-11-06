import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pd
import pwlf
from pwlf import *
import pickle  # 用于存储变量F和T
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号


class Units:
    def __init__(self):
        self.Info = None
        self.Pmax = None
        self.Pmin = None
        self.UT = None
        self.DT = None
        self.tcold = None
        self.hc = None  # 热启动费用
        self.cc = None  # 冷启动费用
        self.sc = None  # 关停费用
        self.RU = None
        self.coeff_1 = None
        self.coeff_2 = None
        self.coeff_3 = None
        self.NL = None
        self.v0 = None  # 初始状态
        self.V0 = None  # 初始开关（01变量）
        self.p0 = None
        self.U0 = None
        self.S0 = None
        self.bus = None

    def read_col(self, recarray, col_name):
        col = recarray[col_name]
        return np.squeeze(col.reshape((len(col)), 1))

    def read_data(self, file_path, sheet_name):

        # 读入文件
        with pd.ExcelFile(file_path) as xls:  # r".\P4参考资料\P4-附件-IEEE-31bus数据.xls"
            df = pd.read_excel(xls, sheet_name)  # 'Sys_ThUnitInfo'
        self.Info = df.to_records(index=False)

        # 读入参数
        self.Pmin = self.read_col(self.Info, '最小出力(MW)')
        self.Pmax = self.read_col(self.Info, '最大出力(MW)')
        self.UT = self.read_col(self.Info, '最小开机时间(h)')
        self.DT = self.read_col(self.Info, '最小关机时间(h)')
        # 冷启动时间：机组从停机状态到到达最小出力所需的时间
        self.tcold = self.read_col(self.Info, '冷启动时间(h)')
        self.hc = self.read_col(self.Info, '热启动费用($)')
        self.cc = self.read_col(self.Info, '冷启动费用($)')
        self.sc = self.read_col(self.Info, '关停费用($)')
        self.RU = self.read_col(self.Info, '爬升速率(MW/h)')  # 过程中的爬升速率
        self.SU = self.read_col(self.Info, '最小出力(MW)')  # 开启时的爬升速率
        self.SD = self.read_col(self.Info, '爬升速率(MW/h)')  # 关停时的下坡速率
        self.RD = self.read_col(self.Info, '爬升速率(MW/h)')  # 过程中的下坡速率
        self.coeff_1 = self.read_col(self.Info, '燃料费用曲线二次项系数')
        self.coeff_2 = self.read_col(self.Info, '燃料费用曲线一次项系数')
        self.coeff_3 = self.read_col(self.Info, '燃料费用曲线常数项')
        self.NL = self.read_col(self.Info, '燃料费用分段数')
        self.v0 = self.read_col(self.Info, '机组初始状态')
        self.p0 = self.read_col(self.Info, '机组初始发电量')
        self.bus = self.read_col(self.Info, '节点编号')
        self.V0 = (self.v0 > 0).astype(int)
        self.U0 = np.where(self.v0 < 0, 0, self.v0)
        self.S0 = np.where(self.v0 > 0, 0, -self.v0)

class Loads:
    def __init__(self):
        self.bus = None
        self.lamd = None
        self.d = None
    def read_col(self, recarray, col_name):
        col = recarray[col_name]
        return np.squeeze(col.reshape((len(col), 1)))
    def read_data(self, file_path, sheet_name):
        # 读入文件
        with pd.ExcelFile(file_path) as xls:
            df = pd.read_excel(xls, sheet_name)
        self.Info = df.to_records(index=False)
        # 读入参数
        self.bus = self.read_col(self.Info, '节点编号')
        self.lamd = self.read_col(self.Info, '负载分配系数')
        # 处理参数 只取大于零的数
        self.bus = self.bus[self.lamd > 0]
        self.lamd = self.lamd[self.lamd > 0]
        # 创建 d 二维数组，用于储存每一时刻每一负载的需求
        self.d = np.zeros(((len(self.lamd)), len(cases.D)))
        for i in range(len(self.lamd)):
            for j in range(len(cases.D)):
                self.d[i,j] = cases.D[j] * self.lamd[i]

class Cases:
    def __init__(self):
        self.Info = None
        self.D = None
        self.R = None

    def read_col(self, recarray, col_name):
        col = recarray[col_name]
        return np.squeeze(col.reshape((len(col), 1)))

    def read_data(self, file_path, sheet_name):
        # 读入文件
        with pd.ExcelFile(file_path) as xls:
            df = pd.read_excel(xls, sheet_name)
        self.Info = df.to_records(index=False)
        # 读入参数
        self.D = self.read_col(self.Info, '系统负荷')
        self.R = self.read_col(self.Info, '系统备用')

class Lines:
    def __init__(self):
        self.F = None
        self.begin_bus = None
        self.end_bus = None

    def read_col(self, recarray, col_name):
        col = recarray[col_name]
        return np.squeeze(col.reshape((len(col), 1)))
    def read_data(self, file_path, sheet_name):
        # 读入文件
        with pd.ExcelFile(file_path) as xls:
            df = pd.read_excel(xls, sheet_name)
        self.Info = df.to_records(index=False)
        self.F = self.read_col(self.Info, '传输线容量(MW)')

def read_I(file_path):
    df = pd.read_excel(file_path, sheet_name='Gama',index_col=0, header= 0)
    global I
    I = df.values

def read(file_path):
    '''
    读取数据
    '''
    units.read_data(file_path, "Sys_ThUnitInfo")
    cases.read_data(file_path, "Case_Info")
    lines.read_data(file_path, "Sys_Line")
    loads.read_data(file_path, "Sys_NetInfo")
    read_I(file_path)



def set_params():
    '''
    设置参数，在字典中调取
    '''
    j = len(units.Info)
    k = len(cases.Info)
    n = len(lines.F)
    m = len(loads.bus)
    params['unit_num'] = j
    params['time_interval'] = k
    params['J'] = range(j)
    params['K'] = range(k)
    params['L'] = range(units.NL[0])
    params['N'] = range(n)
    params['M'] = range(m)


def add_var():
    # 创建一个空字典来储存决策变量对象
    global var_dict
    var_dict = {}
    # shut_down_cost cd_{jk},连续变量
    var_dict['cd'] = model.addVars(
        params['unit_num'], params['time_interval'], vtype=gp.GRB.CONTINUOUS, name='cd')
    # production_cost cp_{jk},连续变量
    var_dict['cp'] = model.addVars(
        params['unit_num'], params['time_interval'], vtype=gp.GRB.CONTINUOUS, name='cp')
    # startup_cost cu_{jk},连续变量
    var_dict['cu'] = model.addVars(
        params['unit_num'], params['time_interval'], vtype=gp.GRB.CONTINUOUS, name='cu')
    # 机组出力p_jk,连续变量
    var_dict['p'] = model.addVars(
        params['unit_num'], params['time_interval'], vtype=gp.GRB.CONTINUOUS, name='p')
    # 机组最大出力pmax_jk,连续变量
    var_dict['pmax'] = model.addVars(
        params['unit_num'], params['time_interval'], vtype=gp.GRB.CONTINUOUS, name='pmax')
    # 机组开关状态v_jk,01变量
    var_dict['v'] = model.addVars(
        params['unit_num'], params['time_interval'], vtype=gp.GRB.BINARY, name='v')
    # delta_{jkl}, 连续变量 power produced in block l of the piecewise linear production cost function of unit j in period k
    var_dict['delta'] = model.addVars(
        params['unit_num'], params['time_interval'], 3, vtype=gp.GRB.CONTINUOUS, name='delta')

    model.update()

    return var_dict


def calc_quadra(a, b, c, x):
    return a * x**2 + b * x + c



def add_constr_objective(var):
    # 读取变量
    cd = var['cd']
    v = var['v']
    cu = var['cu']
    delta = var['delta']
    p = var['p']
    cp = var['cp']
    J = params['J']
    K = params['K']
    L = params['L']
    t = params['time_interval']
    u = params['unit_num']
    C = units.sc
    V0 = units.V0
    S0 = units.S0
    v0 = units.v0
    tcold = units.tcold
    DT = units.DT
    hc = units.hc
    cc = units.cc
    coeff_1 = units.coeff_1
    coeff_2 = units.coeff_2
    coeff_3 = units.coeff_3
    Pmin = units.Pmin
    Pmax = units.Pmax
    NL = units.NL[0]

    # 定义燃料费用函数分段数ND
    ND = np.squeeze(np.full((1, u), t))  # 下标存疑
    # 定义常数字典Kc
    Kc = {}
    # 只计算热启动
    for j in J:
        for k in K:
            Kc[j,t] = hc[j]
    
    # 定义常数数组A(cmin的集合)
    A = calc_quadra(coeff_1, coeff_2, coeff_3, Pmin)

    F=[[10.151950020000989, 11.152000000000006, 12.152049979999006],
         [9.445936025601256, 10.72599999999999, 12.006063974398694],
         [9.445936025601256, 10.72599999999999, 12.006063974398694],
         [9.445936025601256, 10.72599999999999, 12.006063974398694],
         [7.481935026001293, 8.781999999999996, 10.082064973998701],
         [7.481935026001293, 8.781999999999996, 10.082064973998701],
         [10.191960016000797, 10.992000000000003, 11.792039983999208],
         [9.451975010000446, 9.952, 10.452024989999499],
         [9.301975010000472, 9.802000000000012, 10.302024989999504],
         [12.805990503800203, 12.995999999999995, 13.186009496199794],
         [12.123982007200347, 12.484000000000002, 12.84401799279964],
         [10.290966013600665, 10.970999999999993, 11.651033986399316],
         [9.176962515000756, 9.926999999999992, 10.677037484999238],
         [13.308422254433555, 13.530099999999996, 13.751777745566454],
         [14.955988004800243, 15.195999999999998, 15.436011995199756],
         [10.290966013600665, 10.970999999999993, 11.651033986399316]]

    # Star_up Cost 

    model.addConstrs(cu[j, k] >= 0 for j in J for k in K)  # (13)
    model.addConstrs(cu[j, k] >= hc[j]*(v[j, k]-v[j, k-1])
                     for j in J for k in K[1:])  # (15)
    model.addConstrs(cu[j, 0] >= hc[j]*(v[j, 0]-V0[j]) for j in J)
    
    # Production Cost 1152(11294) + 384(11678) + 384(12062) + 384(12446) + 384(12830) + 384(13214)
    model.addConstrs(delta[j, k, l] >=
                     0 for j in J for k in K for l in L)  # (11)

    model.addConstrs(delta[j,k,l] <= (Pmax[j]-Pmin[j])/NL for j in J for k in K for l in range(NL)) #(8)(9)(10)

    model.addConstrs(p[j, k] == gp.quicksum(delta[j, k, l]
                     for l in L) + Pmin[j] * v[j, k] for j in J for k in K)  # (7)

    model.addConstrs(cp[j, k] == A[j] * v[j, k] + gp.quicksum(F[j][l] * delta[j, k, l] for l in L) for j in J for k in K)  # (6)


def add_constr_unit(var):
    # 读取变量
    v = var['v']
    p = var['p']
    pmax = var['pmax']
    J = params['J']
    K = params['K']
    t = params['time_interval']
    u = params['unit_num']
    Pmin = units.Pmin
    Pmax = units.Pmax
    p0 = units.p0
    V0 = units.V0
    U0 = units.U0
    S0 = units.S0
    RU = units.RU
    SU = units.SU
    RD = units.RD
    SD = units.SD
    UT = units.UT
    DT = units.DT

    T = np.squeeze(np.full((1, u), t))
    G = np.minimum(T, (UT - U0) * V0)
    L = np.minimum(T, (DT - S0) * (1 - V0))

    # Generation Limits 1536
    model.addConstrs(p[j, k] >= Pmin[j] * v[j, k]
                     for j in J for k in K)  # (16)
    model.addConstrs(p[j, k] <= pmax[j, k] for j in J for k in K)
    model.addConstrs(pmax[j, k] >= 0 for j in J for k in K)  # (17)
    model.addConstrs(pmax[j, k] <= Pmax[j] * v[j, k] for j in J for k in K)
    # Ramping Constraints 384 + 368 + 384
    model.addConstrs(pmax[j, k] <= p[j, k-1] + RU[j] * v[j, k-1] + SU[j] * (
        v[j, k] - v[j, k-1]) + Pmax[j] * (1 - v[j, k]) for j in J for k in K[1:])  
    model.addConstrs(pmax[j, 0] <= p0[j] + RU[j] * V0[j] + SU[j] * (v[j,0] - V0[j]) + Pmax[j] * (1 - v[j,0]) for j in J) # (18)

    # Minimum Up and Down time Constraints

    # Minimum Up time Constrains (21-23)
    model.addConstrs(gp.quicksum(
        1 - v[j, k] for k in range(G[j])) == 0 for j in J)  # (21)
    model.addConstrs(gp.quicksum(v[j,n] for n in range(UT[j]- 1 + 1)) >= UT[j] * (v[j,0] - V0[j]) for j in J)
    model.addConstrs(gp.quicksum(v[j,n] for n in range(k,k+UT[j]- 1 + 1)) >= UT[j] * (v[j, k] - v[j, k-1]) for j in J for k in range(G[j], t - UT[j] + 1) if k > 0)
    model.addConstrs(gp.quicksum(v[j,n] - (v[j,k] - v[j,k-1]) for n in range(k,t)) >= 0 for j in J for k in range(t-UT[j]+1,t))
    # Minimum Down time Constraints (24-26)
    model.addConstrs(gp.quicksum(v[j, k] for k in range(L[j])) == 0 for j in J)  # (21)
    model.addConstrs(gp.quicksum((1-v[j,n]) for n in range(DT[j]- 1 + 1)) >= DT[j] * (-v[j,0] + V0[j]) for j in J)
    model.addConstrs(gp.quicksum((1-v[j,n]) for n in range(k,k+DT[j]- 1 + 1)) >= DT[j] * (- v[j, k] + v[j, k-1]) for j in J for k in range(L[j], t - DT[j] + 1) if k > 0)
    model.addConstrs(gp.quicksum(1 - v[j,n] - (-v[j,k] + v[j,k-1]) for n in range(k,t)) >= 0 for j in J for k in range(t-DT[j]+1,t))


def add_constr_sys(var):
    # 读取变量
    p = var['p']
    pmax = var['pmax']
    J = params['J']
    K = params['K']
    D = cases.D
    R = cases.R
    # power balance 24 (3464)
    model.addConstrs((gp.quicksum(p[j, k]
                     for j in J) == D[k]) for k in K)  # (18)
    # spinning reserve 24 (3488)
    model.addConstrs(
        (gp.quicksum(pmax[j, k] for j in J) >= D[k] + R[k])for k in K)  # (19)

    return 0

def add_constr_opf(var):
    p = var['p']
    J = params['J']
    K = params['K']
    M = params['M']
    N = params['N']
    
    model.addConstrs(
        (gp.quicksum(I[n, units.bus[j] - 1] * p[j,k] for j in J) - gp.quicksum(I[n, loads.bus[m] - 1] * loads.d[m, k] for m in M) >= -lines.F[n]) for n in N for k in K
    )

    model.addConstrs(
        (gp.quicksum(I[n, units.bus[j] - 1] * p[j,k] for j in J) - gp.quicksum(I[n, loads.bus[m] - 1] * loads.d[m, k] for m in M) <= lines.F[n]) for n in N for k in K
    )

def add_constr(var):

    # 添加与目标函数有关的约束
    add_constr_objective(var)

    # 添加与机组个体有关的约束
    add_constr_unit(var)

    # 添加与机组整体有关的约束
    add_constr_sys(var)

    # 添加与OPF有关的约束

    add_constr_opf(var)

    model.update()


def solve_UC():
    '''
    对UC进行混合整数线性规划求解
    '''
    # 新建决策变量
    var_dict = add_var()
    # 设定目标函数
    cp = var_dict['cp']
    cu = var_dict['cu']
    cd = var_dict['cd']
    J = params['J']
    K = params['K']
    # 设置目标函数，去除了关机费用
    model.setObjective(gp.quicksum(
        cp[j, k] + cu[j, k] for j in J for k in K), gurobipy.GRB.MINIMIZE)
    # 添加约束条件
    add_constr(var_dict)
    num_constraints = len(model.getConstrs())
    print("Number of constraints:", num_constraints)
    model.optimize()


def data_analysis():
    # 提取参数
    J = params['J']
    K = params['K']
    X = range(1, params['time_interval'] + 1)
    p_values = model.getAttr('x', var_dict['p'])
    p_units = []
    p_total = []
    p_accumulate = []
    demand = cases.D

    # set p_total
    for k in K:
        p_total.append(sum(p_values[j, k] for j in J))
    # set p_units
    for j in J:
        p_units.append([p_values[j, k] for k in K])
    p_accumulate.append(p_units[0])
    for j in J[1:]:
        p_accumulate.append(
            [x + y for x, y in zip(p_accumulate[j-1], p_units[j])])

    '''____________每一个机组的发电量____________'''
    for i in range(params['unit_num']):
        x = range(params['time_interval'])
        y = [[p_values[i, j]] for j in x]

        x = range(1, params['time_interval'] + 1)
        plt.plot(x, y, label=f"机组{i+1}")
    plt.title("机组发电量")
    plt.xlabel("时间(h)")
    plt.ylabel("发电量(MW)")
    plt.legend()
    plt.show()
    '''___________累计发电量_____________'''
    for j in J:
        plt.plot(x, p_accumulate[j], label=f"机组{j+1}累积发电量")
    plt.title("机组累积发电量")
    plt.xlabel("时间(h)")
    plt.ylabel("发电量(MW)")
    plt.legend()
    plt.show()
    '''___________总发电量与需求__________'''
    plt.plot(x, p_total, linestyle='solid',
             color='red', marker='o', label=f"总发电量")
    plt.plot(x, demand, linestyle='dashed', color='blue', label=f"总需求")
    plt.title("机组总发电量与总需求关系曲线")
    plt.xlabel("时间(h)")
    plt.ylabel("发电量(MW)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 创建units,cases,lines实例
    units = Units()
    loads = Loads()
    cases = Cases()
    lines = Lines()
    read(r"D:\学习资料\科研\Unit Commitment\P4\P4参考资料\P4-附件-IEEE-31bus数据.xls")
    # 设置params参数列表 params = {'unit_num','time_interval','J','K'}
    params = {}
    set_params()
    # 创建模型，开始优化求解
    model = gp.Model('MILP_UC')
    model.write('UC_MILP.mps')
    solve_UC()
    model.write('UC_MILP_solutions.sol')
    # 根据数据绘制图表
    data_analysis()

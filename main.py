import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pd
import pwlf
from pwlf import * 
import pickle # 用于存储变量F和T
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


class Units:
    def __init__(self):
        self.Info = None
        self.Pmin = None
        self.Pmax = None
        self.UT = None
        self.DT = None
        self.tcold = None
        self.hc = None #热启动费用
        self.cc = None #冷启动费用
        self.sc = None #关停费用
        self.RU = None
        self.coeff_1 = None
        self.coeff_2 = None
        self.coeff_3 = None
        self.NL = None
        self.v0 = None #初始状态
        self.V0 = None #初始开关（01变量）
        self.p0 = None
        self.U0 = None
        self.S0 = None

    def read_col(self,recarray,col_name):
        col = recarray[col_name]
        return np.squeeze(col.reshape((len(col)),1))

    def read_data(self, file_path,sheet_name):

        #读入文件
        with pd.ExcelFile(file_path) as xls: #r".\P4参考资料\P4-附件-IEEE-31bus数据.xls"
            df = pd.read_excel(xls, sheet_name) #'Sys_ThUnitInfo'
        self.Info = df.to_records(index = False)

        #读入参数
        self.Pmin = self.read_col(self.Info, '最小出力(MW)')
        self.Pmax = self.read_col(self.Info, '最大出力(MW)')
        self.UT = self.read_col(self.Info, '最小开机时间(h)')
        self.DT = self.read_col(self.Info, '最小关机时间(h)')
        self.tcold = self.read_col(self.Info, '冷启动时间(h)') # 冷启动时间：机组从停机状态到到达最小出力所需的时间
        self.hc = self.read_col(self.Info, '热启动费用($)') 
        self.cc = self.read_col(self.Info, '冷启动费用($)') 
        self.sc = self.read_col(self.Info, '关停费用($)') 
        self.RU = self.read_col(self.Info, '爬升速率(MW/h)') # 过程中的爬升速率
        self.SU = self.read_col(self.Info, '爬升速率(MW/h)') # 开启时的爬升速率
        self.SD = self.read_col(self.Info, '爬升速率(MW/h)') # 关停时的下坡速率
        self.RD = self.read_col(self.Info, '爬升速率(MW/h)') # 过程中的下坡速率
        self.coeff_1 = self.read_col(self.Info, '燃料费用曲线二次项系数')
        self.coeff_2 = self.read_col(self.Info, '燃料费用曲线一次项系数')
        self.coeff_3 = self.read_col(self.Info, '燃料费用曲线常数项')
        self.NL = self.read_col(self.Info, '燃料费用分段数')
        self.v0 = self.read_col(self.Info, '机组初始状态')
        self.p0 = self.read_col(self.Info, '机组初始发电量')
        self.V0 = (self.v0 > 0).astype(int)
        self.U0 = np.where(self.v0 < 0, 0, self.v0)
        self.S0 = np.where(self.v0 > 0, 0, -self.v0)

class Cases:
    def __init__(self):
        self.Info = None
        self.D = None
        self.R = None
    
    def read_col(self,recarray,col_name):
        col = recarray[col_name]
        return np.squeeze(col.reshape((len(col),1)))
    
    def read_data(self, file_path, sheet_name):
        #读入文件
        with pd.ExcelFile(file_path) as xls:
            df  = pd.read_excel(xls, sheet_name)
        self.Info = df.to_records(index=False)
        #读入参数
        self.D = self.read_col(self.Info,'系统负荷')
        self.R = self.read_col(self.Info,'系统备用')

def read(file_path):
    '''
    读取数据
    '''
    units.read_data(file_path,"Sys_ThUnitInfo")
    cases.read_data(file_path,"Case_Info")

def set_params():
    '''
    设置参数，在字典中调取
    '''
    j = len(units.Info)
    k = len(cases.Info)
    params['unit_num'] = j
    params['time_interval'] = k
    params['J'] = range(j)
    params['K'] = range(k)
    params['L'] = range(units.NL[0])

def add_var():
    # 创建一个空字典来储存决策变量对象
    global var_dict
    var_dict = {}
    # shut_down_cost cd_{jk},连续变量
    var_dict['cd'] = model.addVars(params['unit_num'], params['time_interval'], vtype = gp.GRB.CONTINUOUS, name = 'cd')
    # production_cost cp_{jk},连续变量
    var_dict['cp'] = model.addVars(params['unit_num'], params['time_interval'], vtype = gp.GRB.CONTINUOUS, name = 'cp')
    # startup_cost cu_{jk},连续变量
    var_dict['cu'] = model.addVars(params['unit_num'], params['time_interval'], vtype = gp.GRB.CONTINUOUS, name = 'cu')
    # 机组出力p_jk,连续变量
    var_dict['p'] = model.addVars(params['unit_num'], params['time_interval'], vtype = gp.GRB.CONTINUOUS, name = 'p')
    # 机组最大出力pmax_jk,连续变量
    var_dict['pmax'] = model.addVars(params['unit_num'],params['time_interval'], vtype = gp.GRB.CONTINUOUS, name = 'pmax')
    # 机组开关状态v_jk,01变量
    var_dict['v'] = model.addVars(params['unit_num'], params['time_interval'], vtype = gp.GRB.BINARY, name = 'v')
    # delta_{jkl}, 连续变量 power produced in block l of the piecewise linear production cost function of unit j in period k
    var_dict['delta'] = model.addVars(params['unit_num'], params['time_interval'],3,vtype = gp.GRB.CONTINUOUS, name = 'delta')

    model.update()


    return var_dict

def calc_quadra(a, b, c, x):
    return a * x**2 + b * x + c

def lin_of_quadra():
    pass



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


    # 定义开启费用函数分段数ND
    ND = np.squeeze(np.full((1,u),t)) #下标存疑
    # 定义常数字典Kc
    Kc = {}
    for j in J:
        for t in K:
            if t >= 0 and t <= tcold[j] + DT[j] - 1:
                Kc[j,t] = hc[j]
            elif t >= tcold[j] + DT[j] :
                Kc[j,t] = cc[j]

    # 定义常数数组A(cmin的集合)
    A = calc_quadra(coeff_1, coeff_2, coeff_3, Pmin)
    # # 利用pwlf库，已知分段数，scipy差分进化算法拟合，得到常数字典T与F
    # T = {}
    # F = {}
    # for j in J:
    #     pp = np.linspace(Pmin[j], Pmax[j], num=1000)
    #     cc = calc_quadra(coeff_1[j], coeff_2[j], coeff_3[j], pp)

    #     # 使用pwlf库进行线性拟合
    #     my_pwlf = pwlf.PiecewiseLinFit(pp, cc)
    #     T[j] = my_pwlf.fit(NL)
    #     F[j] = my_pwlf.slopes
    #     values = list(T.values())
    #     T = np.array(values)
    #     T = T[:,1:-1]
    #     values = list(F.values())
    #     F = np.array(values)
    # with open('lin_cons.pkl','wb') as f:
    #     pickle.dump((T, F),f)

    with open('lin_cons.pkl', 'rb') as f:
        T , F = pickle.load(f)
    


   # Shutdown Cost 768
    model.addConstrs(cd[j,k] >= 0 for j in J for k in K) # (14)
    model.addConstrs(cd[j,k] >= C[j]*(v[j,k-1]-v[j,k]) for j in J for k in K[1:]) # (15)
    model.addConstrs(cd[j,0] >= C[j]*(V0[j]-v[j,0]) for j in J)

    # Star_up Cost 384(4640) + 5502(10142 涨得好快！应该是ND设得太大的原因)
    model.addConstrs(cu[j,k] >= 0 for j in J for k in K) #(13)

    for j in J:
        for k in K:
            for t in range(ND[j]):
                if k - t + 1 >= 0 :
                    model.addConstr(cu[j,k] >= Kc[j,t] * (v[j,k] - gp.quicksum(v[j,k-n] for n in range(t))))
                else:
                    if S0[j] >= t - k - 1:
                        model.addConstr(cu[j,k] >= Kc[j,t] * (v[j,k] - gp.quicksum(v[j,k-n] for n in range(k+ 1))))

    # Production Cost 1152(11294) + 384(11678) + 384(12062) + 384(12446) + 384(12830) + 384(13214)
    model.addConstrs(delta[j,k,l] >= 0 for j in J for k in K for l in L) # (11)
    model.addConstrs(delta[j,k,NL-1] <= Pmax[j] - T[j][NL-2] for j in J for k in K) # (10)
    model.addConstrs(delta[j,k,l] <= T[j][l] - T[j][l-1] for j in J for k in K for l in range(1,NL-1)) #(9)
    model.addConstrs(delta[j,k,0] <= T[j][0] - Pmin[j] for j in J for k in K) # (8)
    model.addConstrs(p[j,k] == gp.quicksum(delta[j,k,l] for l in L) + Pmin[j] * v[j,k] for j in J for k in K) # (7)
    
    model.addConstrs(cp[j,k] == A[j] * v[j,k] + gp.quicksum(F[j][l] * delta[j,k,l] for l in L) for j in J for k in K) # (6)

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

    T = np.squeeze(np.full((1,u),t))
    G = np.minimum(T,(UT - U0) * V0)
    L = np.minimum(T,(DT - S0) * (1 - V0))

    
    # Generation Limits 1536
    model.addConstrs(p[j,k] >= Pmin[j] * v[j,k] for j in J for k in K) # (16)
    model.addConstrs(p[j,k] <= pmax[j,k] for j in J for k in K)
    model.addConstrs(pmax[j,k] >= 0 for j in J for k in K) # (17)
    model.addConstrs(pmax[j,k] <= Pmax[j] * v[j,k] for j in J for k in K) 
    # Ramping Constraints 384 + 368 + 384
    model.addConstrs(pmax[j,k] <= p[j,k-1] + RU[j] * v[j,k-1] + SU[j] * (v[j,k] - v[j,k-1]) + Pmax[j] * (1 - v[j,k]) for j in J for k in K[1:]) # (18)
    model.addConstrs(pmax[j,0] <= p0[j] + RU[j] * V0[j] + SU[j] * (v[j,0] - V0[j]) + Pmax[j] * (1 - v[j,0])for j in J)
    model.addConstrs(pmax[j,k] <= Pmax[j] * v[j,k+1] + SD[j] * (v[j,k] - v[j,k+1]) for j in J for k in K[:-1]) # (19)
    model.addConstrs(p[j,k-1] - p[j,k] <= RD[j] * v[j,k] + SD[j] * (v[j,k-1] - v[j,k]) + Pmax[j] * (1 - v[j,k-1]) for j in J for k in K[1:]) # (20)
    model.addConstrs(p0[j] - p[j,0] <= RD[j] * v[j,0] + SD[j] * (V0[j] - v[j,0]) + Pmax[j] * (1 - V0[j]) for j in J)
    # Minimum Up and Down time Constraints 384(3872) + 384(4256)
    # Minimum Up time Constrains
    model.addConstrs(gp.quicksum(1 - v[j,k] for k in range(G[j])) == 0 for j in J if G[j] != 0) # (21)
    for j in J: # (22)
        model.addConstrs((gp.quicksum(v[j,n] for n in range(k, k + UT[j])) >= UT[j] * (v[j,k] - v[j,k-1]) for k in range(G[j] + 1, t - UT[j] + 1)))
        if (G[j] != 0):
            model.addConstrs((gp.quicksum(v[j,n] for n in range(k, k + UT[j])) >= UT[j] * (v[j,k] - v[j,k-1]) for k in range(G[j], G[j] + 1)))
        else:
            model.addConstrs((gp.quicksum(v[j,n] for n in range(k, k + UT[j])) >= UT[j] * (v[j,k] - V0[j]) for k in range(G[j], G[j] + 1)))
    model.addConstrs(gp.quicksum(v[j,n] - (v[j,k] - v[j,k-1]) for n in range(k, t)) >= 0 for j in J for k in range(t - UT[j] + 1, t)) # (23)
    # Minimum Down time Constraints
    model.addConstrs(gp.quicksum(v[j,k] for k in range(L[j])) == 0 for j in J if L[j] != 0) # (24)
    for j in J: # (25)
        model.addConstrs((gp.quicksum((1 - v[j,n]) for n in range(k, k + DT[j])) >= DT[j] * (v[j,k-1] - v[j,k]) for k in range(L[j] + 1, t - DT[j] + 1)))
        if (L[j] != 0):
            model.addConstrs((gp.quicksum((1 - v[j,n]) for n in range(k, k + DT[j])) >= DT[j] * (v[j,k - 1] - v[j,k]) for k in range(L[j], L[j] + 1)))
        else:
            model.addConstrs((gp.quicksum((1 - v[j,n]) for n in range(k, k + DT[j])) >= DT[j] * (V0[j] - v[j,k]) for k in range(L[j], L[j] + 1)))
    model.addConstrs(gp.quicksum(1 - v[j,n] - (v[j,k-1] - v[j,k]) for n in range(k, t)) >= 0 for j in J for k in range(t - DT[j] + 1, t)) # (26)

def add_constr_sys(var):
    # 读取变量
    p = var['p']
    pmax = var['pmax']
    J = params['J']
    K = params['K']
    D = cases.D
    R = cases.R
    # power balance 24 (3464)
    model.addConstrs((gp.quicksum(p[j,k] for j in J) == D[k]) for k in K) # (18)
    # spinning reserve 24 (3488)
    model.addConstrs((gp.quicksum(pmax[j,k] for j in J) >= D[k] + R[k])for k in K) # (19)


    return 0
def add_constr(var):

    # 添加与目标函数有关的约束
    add_constr_objective(var)

    # 添加与机组个体有关的约束
    add_constr_unit(var)

    # 添加与机组整体有关的约束
    add_constr_sys(var)
 
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
    model.setObjective(gp.quicksum(cp[j,k] + cu[j,k] + cd[j,k] for j in J for k in K) ,gurobipy.GRB.MINIMIZE)
    # 添加约束条件
    add_constr(var_dict)
    model.optimize()

def data_analysis():
    #提取参数
    J = params['J']
    K = params['K']
    X = range(1,params['time_interval'] + 1)
    p_values = model.getAttr('x', var_dict['p'])
    p_units = []
    p_total = []
    p_accumulate = []
    demand = cases.D

    # set p_total
    for k in K:
        p_total.append(sum(p_values[j,k] for j in J))
    # set p_units
    for j in J:
        p_units.append([p_values[j,k] for k in K])
    p_accumulate.append(p_units[0])
    for j in J[1:]:
        p_accumulate.append([x + y for x, y in zip(p_accumulate[j-1], p_units[j])])

    '''____________每一个机组的发电量____________'''
    for i in range(params['unit_num']):
        x = range(params['time_interval'])
        y = [[p_values[i,j]] for j in x]
         
        x = range(1,params['time_interval'] + 1)
        plt.plot(x,y,label=f"机组{i+1}")
    plt.title("机组发电量")
    plt.xlabel("时间(h)")
    plt.ylabel("发电量(MW)")
    plt.legend()
    plt.show()
    '''___________累计发电量_____________'''
    for j in J:
        plt.plot(x,p_accumulate[j],label=f"机组{j+1}累积发电量")
    plt.title("机组累积发电量")
    plt.xlabel("时间(h)")
    plt.ylabel("发电量(MW)")
    plt.legend()
    plt.show()
    '''___________总发电量与需求__________'''
    plt.plot(x,p_total,linestyle='solid', color='red', marker='o',label=f"总发电量")
    plt.plot(x,demand,linestyle='dashed', color='blue',label=f"总需求")
    plt.title("机组总发电量与总需求关系曲线")
    plt.xlabel("时间(h)")
    plt.ylabel("发电量(MW)")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # 创建units,cases实例
    units = Units() 
    cases = Cases()
    read(r".\P4参考资料\P4-附件-IEEE-31bus数据.xls")
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
    






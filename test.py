import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pd

model = gp.Model('LP_test')

x = model.addVar(vtype = gp.GRB.CONTINUOUS)

model.addConstr(x <= 5)

model.update()

print(model.NumConstrs)
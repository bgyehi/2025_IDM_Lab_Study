import gurobipy as gp
from gurobipy import GRB

# 모델 생성
m = gp.Model("example")

# 변수 생성
x = m.addVar(vtype=GRB.BINARY, name="x")
y = m.addVar(vtype=GRB.BINARY, name="y")

# 목적함수 설정
m.setObjective(x + y, GRB.MAXIMIZE)

# 제약조건 추가
m.addConstr(x + 2*y <= 1, "c0")

# 최적화
m.optimize()

# 결과 출력
for v in m.getVars():
    print(v.varName, v.x)

print("Obj:", m.objVal)



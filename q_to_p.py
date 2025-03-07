import re
import numpy as np
from sympy.parsing.mathematica import parse_mathematica
from sympy import var
beta,lmb = var('beta lmb')
expr='-((lmb Sqrt[q1/q0] (lmb^2 (-1+2 lmb) (1+b (1+lmb) (-1+q0))+lmb^2 (-1+2 lmb) (1+b (1+lmb) (-1+q1))-(lmb^2-(1-lmb-b (-1+lmb+2 lmb^2) (-1+q0)) (1-lmb-b (-1+lmb+2 lmb^2) (-1+q1))) (1-lmb-b (-1+lmb+2 lmb^2) (-1+q2)))^2+b (-1+lmb+2 lmb^2)^3 q0 (lmb^2-(1-lmb-b (-1+lmb+2 lmb^2) (-1+q1)) (1-lmb-b (-1+lmb+2 lmb^2) (-1+q2)))^2-2 b lmb (-1+2 lmb) (-1+lmb+2 lmb^2)^3 Sqrt[q0 q1] (lmb^2-(1-lmb-b (-1+lmb+2 lmb^2) (-1+q1)) (1-lmb-b (-1+lmb+2 lmb^2) (-1+q2))) (1+b (1+lmb) (-1+q2))+b (1-2 lmb)^2 lmb^2 (-1+lmb+2 lmb^2)^3 q1 (1+b (1+lmb) (-1+q2))^2+b (1-2 lmb)^2 lmb^2 (-1+lmb+2 lmb^2)^3 (1+b (1+lmb) (-1+q1))^2 q2+lmb (lmb^2 (-1+2 lmb) (1+b (1+lmb) (-1+q0))+lmb^2 (-1+2 lmb) (1+b (1+lmb) (-1+q1))-(lmb^2-(1-lmb-b (-1+lmb+2 lmb^2) (-1+q0)) (1-lmb-b (-1+lmb+2 lmb^2) (-1+q1))) (1-lmb-b (-1+lmb+2 lmb^2) (-1+q2)))^2 Sqrt[q2/q0]-2 b lmb (-1+2 lmb) (-1+lmb+2 lmb^2)^3 (1+b (1+lmb) (-1+q1)) (lmb^2-(1-lmb-b (-1+lmb+2 lmb^2) (-1+q1)) (1-lmb-b (-1+lmb+2 lmb^2) (-1+q2))) Sqrt[q0 q2]+2 b (1-2 lmb)^2 lmb^2 (-1+lmb+2 lmb^2)^3 (1+b (1+lmb) (-1+q1)) (1+b (1+lmb) (-1+q2)) Sqrt[q1 q2])/((1-lmb-2 lmb^2) (lmb^2 (-1+2 lmb) (1+b (1+lmb) (-1+q0))+lmb^2 (-1+2 lmb) (1+b (1+lmb) (-1+q1))-(lmb^2-(1-lmb-b (-1+lmb+2 lmb^2) (-1+q0)) (1-lmb-b (-1+lmb+2 lmb^2) (-1+q1))) (1-lmb-b (-1+lmb+2 lmb^2) (-1+q2)))^2))'
print(parse_mathematica(expr))


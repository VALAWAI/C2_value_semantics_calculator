import valawex
from sympy import Add, symbols

crit_symbols  = symbols('c(1:16)')
action_symbols  = symbols('a(1:11)')
postcrit_symbols  = symbols('pc(1:16)')

print(crit_symbols[13], action_symbols[9])


expr = Add(crit_symbols[1], postcrit_symbols[2], action_symbols[2])

print(expr)

p1 = valawex.analysis.Patient([1, 2, 2],[0, 1, 0, 0], [2, 3, 5])

print(p1.get_criteria())

print(p1.alignment_evaluation(expr))


p2 = valawex.analysis.Patient()

print(p2.get_criteria())
print(p2.get_actions())
print(p2.get_alignment())


from sympy import (
    S,
    Gt,
    log,
    Pow,
    limit,
    Symbol,
    solveset,
    Intersection,
    singularities,
    simplify,
)
from sympy.plotting import plot
from sympy.parsing.latex import parse_latex

import csv
from typing import Tuple, Union, Type


class T:
    class IsEvenFunc:
        pass

    class IsOddFunc:
        pass


class Expression:
    def __init__(self, expression: Pow, var: Symbol):
        self.expr = expression
        self.var = var

    def __str__(self):
        return str(self.expr)

    def FunctionValue(self, point: float) -> float:
        return self.expr.subs(self.var, point)

    def FindDiscontinuity(self):
        special_points = singularities(self.expr, self.var)
        if special_points:
            return special_points
        return None

    def FindDomain(self) -> Union[S.Reals, Tuple]:
        discontinuity = self.FindDiscontinuity()
        if not discontinuity:
            return S.Reals
        domain = S.Reals - discontinuity
        if self.expr.has(log):
            for arg in self.expr.args:
                if arg.has(log):
                    log_domain = solveset(Gt(arg.args[0], 0), self.var, domain=S.Reals)
                    domain = Intersection(domain, log_domain)

        return domain

    def FindSymmetry(self) -> Type[T.IsEvenFunc] | Type[T.IsOddFunc] | None:
        if self.expr.subs(self.var, -self.var) == self.expr:
            return T.IsEvenFunc
        elif self.expr.subs(self.var, -self.var) == -self.expr:
            return T.IsOddFunc
        return None

    def Derivative(self) -> Pow:
        return self.expr.diff(self.var)

    def FindCriticalPoints(self):
        potential_local_extremum = solveset(self.Derivative(), self.var)
        critical_points = []
        for point in potential_local_extremum:
            if self.SecondDerivative().subs(self.var, point) != 0:
                critical_points.append(point)
        return critical_points

    def SecondDerivative(self) -> Pow:
        return self.Derivative().diff(self.var)

    def ThirdDerivative(self) -> Pow:
        return self.SecondDerivative().diff(self.var)

    def FindInflectionPoints(self):
        potential_inflection_points = solveset(self.SecondDerivative(), self.var)
        inflection_points = []

        for point in potential_inflection_points:
            third_derivative_value = self.ThirdDerivative().subs(self.var, point)

            if third_derivative_value != 0:
                inflection_points.append(point)

        return inflection_points


x = Symbol("x")
latex_str = "\\frac{1}{x} + \\ln x"
expr = Expression(parse_latex(latex_str), x)

print(f"정의역 : {(expr.FindDomain())}")

print(
    f'대칭성 : {"우함수(y축 대칭)" if expr.FindSymmetry() == T.IsEvenFunc else "기함수(원점 대칭)" if expr.FindSymmetry() == T.IsOddFunc else "없음"}'
)
print(f"도함수 : {simplify(expr.Derivative())}")

print(f"이계도함수 : {simplify(expr.SecondDerivative())}")

print("극점 : ")
print(
    "\n".join(
        [
            f"({str(point)}, {expr.FunctionValue(point)})"
            for point in expr.FindCriticalPoints()
        ]
    )
)

print("변곡점 : ")
print(
    "\n".join(
        [
            f"({str(point)}, {expr.FunctionValue(point)})"
            for point in expr.FindInflectionPoints()
        ]
    )
)

print(f"특이점 : {', '.join([f'x = {p}' for p in expr.FindDiscontinuity()])}")

displayed_points = []
for points in [
    expr.FindCriticalPoints(),
    expr.FindInflectionPoints(),
    expr.FindDiscontinuity(),
]:
    for point in points:
        displayed_points.append(point)
displayed_points.sort()

response = [["x"], ["f(x)"], ["f'(x)"], ["f''(x)"]]

for point in displayed_points:
    if point not in expr.FindDomain():
        response[0].append(f"'({str(point)})")
        response[1].append("")
        response[2].append("")
        response[3].append("")
    else:
        response[0].append(str(point))
        response[1].append(str(expr.FunctionValue(point)))
        response[2].append(str(expr.Derivative().subs(x, point)))
        response[3].append(str(expr.SecondDerivative().subs(x, point)))
    response[0].append("...")
    response[1].append("...")
    if point not in expr.FindDomain():
        response[2].append("...")
    else:
        if expr.SecondDerivative().subs(x, point) > 0:
            response[2].append("+")
        elif expr.SecondDerivative().subs(x, point) < 0:
            response[2].append("-")
        else:
            response[2].append("...")
    response[3].append("...")


with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerows(response)

plot(expr.expr, (x, 0.1, 10), show=False).save("plot.png")

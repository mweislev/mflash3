import numpy as np
import numbers
import operator as op
import warnings

__author__ = "Michael Weis"
__version__ = "1.0.0.0"

upn_binop = {"+" : op.add, "-" : op.sub, "*" : op.mul, "/" : op.truediv,
    "^" : op.pow, "**" : op.pow, "|": op.or_, "&": op.and_, "//": op.floordiv,
    "<": op.lt, "<=": op.le, "==": op.eq, "!=": op.ne, ">=": op.ge, ">": op.gt}
    
# The following list is needed for checking unhashable datatypes against the
# list of operators. While operator strings itself are always hashable,
# not all stack items are (e.g. upn lists themselves).
upn_binop_keys = list(upn_binop.keys())

class upn(list):
    def __init__(self, upn_list, str_eval=eval):
        self.__str_eval = str_eval
        list.__init__(self, upn_list)
    def eval(self):
        stack = []
        while self:
            item = self.pop(0)
            if isinstance(item, numbers.Number) or isinstance(item, np.ndarray):
                stack.append(item)
            elif item in upn_binop_keys:
                b = stack.pop()
                a = stack.pop()
                r = upn_binop[item](a, b)
                self.insert(0, r)
            elif isinstance(item, list):
                r = upn(item).eval()
                self.insert(0, r)
            elif isinstance(item, np.ufunc):
                args = []
                for i in range(item.nin):
                    args.insert(0, stack.pop())
                r = item(*args)
                if isinstance(r, list):
                    self.insert(0, upn(r).eval())
                else:
                    self.insert(0, r)
            elif callable(item):
                a = stack.pop()
                r = item(a)
                self.insert(0, r)
            elif isinstance(item, str) or isinstance(item, unicode):
                r = self.__str_eval(item)
                self.insert(0, r)
            else:
                warnings.warn('Alien UPN input: "'+str(item)+'"', RuntimeWarning)
                stack.append(item)
        self.extend(stack)
        return stack[0] if len(stack)==1 else upn(stack)

# ==== TEST ====================================================================
if __name__ == '__main__':
    a1 = np.array((1,2,3))
    l1 = (a1, 2, '**')
    p1 = upn(l1)
    print(p1.eval())
    p2a = upn((p1, 3, '+'))
    p2b = upn((upn(l1), 3, '+'))
    print(p2a.eval(), p2b.eval())
    p3 = upn((2, 7, np.power))
    print(p3.eval())
    f = lambda z: lambda y: lambda x: x*y*z
    p4 = upn((5, 7, 11, f))
    print(p4.eval())

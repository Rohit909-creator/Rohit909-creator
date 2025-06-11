'''
Vector for vector based calculations


'''

import math

class vector():

    def __init__(self,a,b):
        self.a = a
        self.b = b

    def __repr__(self):
        return "Vector(%s,%r)" % (self.a,self.b)

    def __abs__(self):
        return math.sqrt(self.a**2 + self.b**2)
    def __mul__(self,X):
        return Vector(self.a * X.a,self.b * X.b)
    def __add__(self,X):
        return Vector(self.a + X.a,self.b + X.b)

    
if __name__ == "__main__":
    v = vector(1,2)

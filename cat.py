from dataclasses import dataclass
from typing import Callable

# Define to avoid warnings
Morphism = None

@dataclass
class CatObject:
    sym: str
    
    def id(self) -> Morphism:
        return Morphism(
            sym='m' + self.sym,
            dom=self,
            cod=self
        )

@dataclass
class Morphism:
    sym: str
    dom: CatObject
    cod: CatObject
    
    def __str__(self):
        return f"{self.sym}: {self.dom.sym} -> {self.cod.sym}"
    
    def compose(self, other):
        if self.cod != other.dom:
            raise TypeError(f"{self.cod} must be equal to {other.dom}")
        return Morphism(
            sym=self.sym + other.sym,
            dom=self.dom,
            cod=other.cod
        )

@dataclass
class Category:
    ob: Callable[[any], bool]
    mor: Callable[[any], bool]




oa = CatObject('A')
ob = CatObject('B')
oc = CatObject('C')
od = CatObject('D')
oa.id()
ob.id()
m1 = Morphism('f', oa, ob)
m2 = Morphism('g', ob, oc)
m3 = Morphism('g', oc, od)
m1

c = Category(
    ob=lambda x: x in [oa, ob],
    mor=lambda x: x in [m1, oa.id(), ob.id()]
)

# oa.id().compose(ob.id()) # raises an error
m1.compose(m2) # fg: A -> C

# Demonstrating associativity
(m1.compose(m2)).compose(m3) # fgg: A -> D
m1.compose(m2.compose(m3)) # fgg: A -> D

c.ob(oc)
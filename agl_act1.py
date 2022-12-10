
from fuzzylogic.classes import Domain, Set, Rule
from fuzzylogic.hedges import very
from fuzzylogic.functions import R, S
from matplotlib import pyplot
import numpy as np

drbly = Domain("Durability", 0, 10)
mlg = Domain("Gas Mileage", 0, 50)
wrth = Domain("Worth", 0, 100)

drbly.weak = S(2,6)
drbly.strong = R(5,9)
drbly.weak.plot()
drbly.strong.plot()

mlg.low = S(5,17)
mlg.high = R(17,45)
mlg.low.plot()
mlg.high.plot()

wrth.worthwile = R(60,100)
wrth.scam = ~wrth.worthwile
wrth.scam.plot()
wrth.worthwile.plot()

R1 = Rule({(drbly.weak, mlg.low): very(wrth.scam)})
R2 = Rule({(drbly.strong, mlg.low): wrth.scam})
R3 = Rule({(drbly.weak, mlg.high): wrth.worthwile})
R4 = Rule({(drbly.strong, mlg.high): very(wrth.worthwile)})

rules = Rule({(drbly.weak, mlg.low): very(wrth.scam),
              (drbly.strong, mlg.low): wrth.scam,
              (drbly.weak, mlg.high): wrth.worthwile,
              (drbly.strong, mlg.high): very(wrth.worthwile),
             })

rules == R1 | R2 | R3 | R4 == sum([R1, R2, R3, R4])

values = {drbly: 8, mlg: 43}

print(R1(values), R2(values), R3(values), R4(values), "=>", rules(values))


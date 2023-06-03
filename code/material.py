# --------------- Material Object --------------- #
# Parameters:                                     #
#       - name = name of material (string)        #
#       - k = thermal conductivity                #
#       - rho = density                           #
#       - c = Coefficient of heat                 #
#       - maxTempt = Max allowable temperature    #
# ----------------------------------------------- #

class Material():
    def __init__(self, name, k, rho, c, maxTemp, cost):
        self.name = name
        self.k = k
        self.rho = rho
        self.c = c
        self.maxTemp = maxTemp
        self.cost = cost
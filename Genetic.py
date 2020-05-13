##Optimization Genetic Approch
import math as mat
import numpy as np
import time
from numpy.linalg import norm
from numpy.random import randint, rand
from MPCMain import MPC, Order


##resulotion = 1/(2^n-1)*range ---> the resulotion of the argument
##Generation ---> number of optimization iterations
##controlarguments ---> the number of control efforts type
##fitness ---> the probability of the candidate to be choosen
##Code ---> the binary code for the candidate


class Candidate:
    def __init__(self):
        self.fitness = []
        self.relative_fitness = []
        self.Target_Value = []
        self.Generation = []
        self.Code = 0
        self.Value = []

    @property
    def Code(self):
        return self.__Code

    @Code.setter
    def Code(self, Code):
        self.__Code = np.asarray(Code)


class DNA(Candidate):
    def __init__(
        self, Number_of_Candidate, Val_max, Val_min, resulotion, controlarguments,
    ):
        super().__init__()
        self.iteration = 0
        self.Number_of_Candidate = Number_of_Candidate
        self.Candidate_List = []
        self.Val_max = Val_max
        self.Val_min = Val_min
        self.Resulotion_init = resulotion
        self.Resulotion = []
        self.Controlarguments = controlarguments
        self.argument_bits = []
        self.NumberofBits = 0
        self.Fitsum = 0
        self.Parent_List = []

    @property
    def Number_of_Candidate(self):
        return self.__Number_of_Candidate

    @property
    def NumberofBits(self):
        return self.__NumberofBits

    @Number_of_Candidate.setter
    def Number_of_Candidate(self, Number_of_Candidate):
        self.__Number_of_Candidate = Number_of_Candidate

    @NumberofBits.setter
    def NumberofBits(self, NumberofBits):
        self.__NumberofBits = NumberofBits

    def initial_Parent_List(self):
        self.Parent_List = np.zeros([self.Number_of_Candidate, self.NumberofBits])

    def Calculate_NumberofBits(self):
        for i in range(self.Controlarguments):
            self.NumberofBits = self.NumberofBits + mat.ceil(
                (
                    mat.log2(
                        np.abs(self.Val_max[i] - self.Val_min[i] + self.Resulotion_init)
                        / self.Resulotion_init
                    )
                )
            )
            self.argument_bits.append(
                mat.ceil(
                    (
                        mat.log2(
                            np.abs(
                                self.Val_max[i] - self.Val_min[i] + self.Resulotion_init
                            )
                            / self.Resulotion_init
                        )
                    )
                )
            )
            self.Resulotion.append(
                (self.Val_max[i] - self.Val_min[i]) / (2 ** (self.argument_bits[i]) - 1)
            )

    def Initialize_Population(self):
        for i in range(self.Number_of_Candidate):
            self.Candidate_List.append(Candidate())
            self.Candidate_List[i].Code = self.Code_init()
            self.Candidate_List[i].Value = self.Calculate_Value(
                self.Candidate_List[i].Code,
                self.Controlarguments,
                self.argument_bits,
                self.Resulotion,
                self.Val_min,
            )
            while not self.Constraint(self.Candidate_List[i].Value):
                self.Candidate_List[i].Code = self.Code_init()
                self.Candidate_List[i].Value = self.Calculate_Value(
                    self.Candidate_List[i].Code,
                    self.Controlarguments,
                    self.argument_bits,
                    self.Resulotion,
                    self.Val_min,
                )

    def Code_init(self):
        Code = []
        for _ in range(self.NumberofBits):
            Code.append(randint(2))
        return Code

    def DNA_fitness(self):
        M_value = 0
        Max = 0
        self.Fitsum = 0
        for i in range(self.Number_of_Candidate):
            self.Candidate_List[i].Target_Value = self.Calculate_Target(
                self.Candidate_List[i].Value
            )
            if i == 0 or M_value > self.Candidate_List[i].Target_Value:
                M_value = self.Candidate_List[i].Target_Value
        if M_value < 0:
            for i in range(self.Number_of_Candidate):
                self.Candidate_List[i].fitness = 1 / (
                    1 + self.Candidate_List[i].Target_Value - M_value
                )
                self.Fitsum = self.Fitsum + self.Candidate_List[i].fitness
                if i == 0 or Max < self.Candidate_List[i].fitness:
                    j = i
                    Max = self.Candidate_List[i].fitness
        else:
            for i in range(self.Number_of_Candidate):
                self.Candidate_List[i].fitness = 1 / (
                    1 + self.Candidate_List[i].Target_Value
                )
                self.Fitsum = self.Fitsum + self.Candidate_List[i].fitness
                if i == 0 or Max < self.Candidate_List[i].fitness:
                    j = i
                    Max = self.Candidate_List[i].fitness
        self.Parent_List[0] = self.Candidate_List[j].Code

    def Parent_Update(self):
        val = 0
        for i in range(self.Number_of_Candidate):
            self.Candidate_List[i].relative_fitness = (
                self.Candidate_List[i].fitness / self.Fitsum
            ) + val
            val = self.Candidate_List[i].relative_fitness
        for i in range(1, self.Number_of_Candidate):
            val = rand()
            j = 0
            while self.Candidate_List[j].relative_fitness < val:
                j = j + 1
            self.Parent_List[i] = self.Candidate_List[j].Code

    def CroosoverandMutation(self):
        # TODO: ID Check
        self.Candidate_List[0].Code = self.Parent_List[0]
        self.Candidate_List[0].Value = self.Calculate_Value(
            self.Candidate_List[0].Code,
            self.Controlarguments,
            self.argument_bits,
            self.Resulotion,
            self.Val_min,
        )
        for i in range(1, self.Number_of_Candidate):
            val = randint(9)
            num1 = int(randint(self.Number_of_Candidate))
            num2 = int(randint(self.Number_of_Candidate))
            if val < 7:
                V = randint(np.sum(self.argument_bits))
                self.Candidate_List[i].Code = np.concatenate(
                    (self.Parent_List[num1][:V], self.Parent_List[num2][V:]), axis=None,
                )
                self.Candidate_List[i].Value = self.Calculate_Value(
                    self.Candidate_List[i].Code,
                    self.Controlarguments,
                    self.argument_bits,
                    self.Resulotion,
                    self.Val_min,
                )
                while not self.Constraint(self.Candidate_List[i].Value):
                    V = randint(self.NumberofBits)
                    self.Candidate_List[i].Code = np.concatenate(
                        (self.Parent_List[num1, :V], self.Parent_List[num2, V:],),
                        axis=None,
                    )
                    self.Candidate_List[i].Value = self.Calculate_Value(
                        self.Candidate_List[i].Code,
                        self.Controlarguments,
                        self.argument_bits,
                        self.Resulotion,
                        self.Val_min,
                    )
            else:
                V = randint(np.sum(self.argument_bits))
                self.Candidate_List[i].Code = self.Parent_List[num1]
                if self.Candidate_List[i].Code[V]:
                    self.Candidate_List[i].Code[V] = 0
                else:
                    self.Candidate_List[i].Code[V] = 1
                self.Candidate_List[i].Value = self.Calculate_Value(
                    self.Candidate_List[i].Code,
                    self.Controlarguments,
                    self.argument_bits,
                    self.Resulotion,
                    self.Val_min,
                )
                while not self.Constraint(self.Candidate_List[i].Value):
                    if self.Candidate_List[i].Code[V]:
                        self.Candidate_List[i].Code[V] = 0
                    else:
                        self.Candidate_List[i].Code[V] = 1
                    V = randint(np.sum(self.argument_bits))
                    if self.Candidate_List[i].Code[V]:
                        self.Candidate_List[i].Code[V] = 0
                    else:
                        self.Candidate_List[i].Code[V] = 1
                    self.Candidate_List[i].Value = self.Calculate_Value(
                        self.Candidate_List[i].Code,
                        self.Controlarguments,
                        self.argument_bits,
                        self.Resulotion,
                        self.Val_min,
                    )

    @staticmethod
    def Calculate_Value(Code, Number_of_arguments, argument_bits, Resulotion, Val_min):
        Value = np.zeros([Number_of_arguments,])
        Running = 0
        for i in range(Number_of_arguments):
            Multi = 2 ** (argument_bits[i] - 1)
            for j in range(Running, argument_bits[i] + Running):
                Value[i] = Value[i] + Code[j] * Multi
                Multi = Multi / 2
            Running = Running + argument_bits[i]
            Value[i] = (Value[i]) * Resulotion[i] + Val_min[i]
        return Value

    # @staticmethod
    # def Constraint(Control_efforts):
    #     if norm(Control_efforts) > 8:
    #         return 1
    #     else:
    #         return 0

    # @staticmethod
    # def Calculate_Target(Control_efforts):
    #     return norm(Control_efforts) ** 2 + Control_efforts[1] ** 3


##RunningFunciton
Rmax = 0.5
Initial_Position = np.zeros([5, 1])
# Optimal_path = ......!!
# For first running we will check Path_center = Optimal_path
Optimal_path = np.array([[5, 10, 15, 20], [5, 10, 15, 20]])
Path_center = Optimal_path
Weights = np.ones([6, 1])
Time_Delta = 1
##Genetic initialization
Number_of_Candidate = 10
# control efforts
Val_max = np.array([4, mat.pi / 4, 4, mat.pi / 4, 4, mat.pi / 4])
Val_min = np.array([-2, -mat.pi / 4, -2, -mat.pi / 4, -2, -mat.pi / 4])
resulotion = 0.01
# control arguments gas and steering
controlarguments = 6

K = DNA(Number_of_Candidate, Val_max, Val_min, resulotion, controlarguments,)
K2 = MPC(Rmax, Initial_Position, Optimal_path, Path_center, Weights, Time_Delta)
K.Calculate_NumberofBits()
K.initial_Parent_List()
K.Initialize_Population()
K.DNA_fitness()
K.CroosoverandMutation()
for i in range(100):
    print(K.Candidate_List[0].Target_Value)
    print(K.Candidate_List[0].Value)
    K.DNA_fitness()
    K.Parent_Update()
    K.CroosoverandMutation()

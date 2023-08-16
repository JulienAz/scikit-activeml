from collections import deque

import numpy as np

from skactiveml.base import BudgetManager


class EstimatedBudgetXu(BudgetManager):
    def __init__(self, budget=None, w=300, theta=1.0, s=0.01):
        super().__init__(budget)
        self.w = w
        self.theta = theta
        self.label_spending = deque(maxlen=w)
        self.s = s

    def query_by_random(self, random_variable):
        queried_indices = []

        #Check if budget left
        if sum(self.label_spending)/self.w < self.budget:
            if random_variable <= self.theta:
                queried_indices.append(0)
                self.label_spending.append(1)
            else:
                self.label_spending.append(0)
        else:
            self.label_spending.append(0)
        return queried_indices

    def query_by_utility(self, uncertainty):
        queried_indices = []
        if sum(self.label_spending)/self.w < self.budget:
            if uncertainty < self.theta:
                self.theta =  self.theta * (1 - self.s)
                queried_indices.append(0)
                self.label_spending.append(1)
            else:
                self.theta = self.theta * (1 + self.s)
                self.label_spending.append(0)
        else:
            self.label_spending.append(0)
        return queried_indices

    def update(self, candidates, queried_indices, *args, **kwargs):
        return self
class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        weight = init # weight is x

        for iteration in range(iterations):
            derivative = 2 * weight
            weight = weight - derivative * learning_rate

        return round(weight, 5)
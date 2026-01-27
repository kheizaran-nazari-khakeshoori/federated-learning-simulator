import numpy as np
def kd_loss(student, teacher, T=2):
    return ((student-teacher)**2).mean()

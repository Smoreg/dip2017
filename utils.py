import numpy as np


def generate_summands(num_ran=[2, 10], num_sum=5, num_width=6):
    vars_names = np.arange(*num_ran)
    vars_names = np.r_[vars_names, np.zeros_like(vars_names)]
    res = np.array([np.random.choice(vars_names, num_width, replace=0)])
    res[0].sort()
    for _ in range(num_sum):
        for i in range(1000):
            applicant = np.array([np.random.choice(vars_names, num_width, replace=0)])
            applicant.sort()
            if not np.any(~np.any(res - applicant, axis=1)) and (not np.all(applicant == 0)):
                break
        else:
            raise Exception('cant make impls')
        res = np.r_[res, applicant]
    print(repr(res))

from itertools import product, groupby

import numpy as np

DEBUG = True


class CNF_builder:
    """
    Класс для превращения полиномов в окломинимальные КНФ и получения их статистики
    """
    def __init__(self, cipher):
        self.cipher = cipher

    class TruthTable:
        """
        TrTab from poly
        tt_left   tt_right
        x y        x ^ y
        0 0        0
        0 1        1
        1 0        1
        1 1        0

        """

        def __init__(self, inp_poly):

            self.form = inp_poly.form
            self.const = inp_poly.const

            vars_nums = np.unique(self.form)
            vars_nums = vars_nums[vars_nums > 1]
            vars_len = len(vars_nums)
            if DEBUG and vars_len > 15:
                raise self.TrTabException('Too many vars')

            self.tt_left = np.array(list(product((1, 0), repeat=vars_len)), dtype=np.bool)
            self.tt_left = np.flipud(self.tt_left)  # make tt_left[0] = [False, ... False]
            res = []
            for i in self.tt_left:
                res.append(inp_poly.solve_poly(vars_nums[i]))
            self.tt_right = np.array(res, dtype=np.bool)

        class TrTabException(Exception):
            pass

    def _grouper(self, summands):
        """
        Группирует суманды
        в группы с числом уникальных перменных меньше либо равных T
        :param summands: Массив чисел
        :param T: максимальный deg в сумманде
        :return: номера групп для суммандов
        """

        groups = np.arange(len(summands))
        T = self.cipher.T
        # Merge all
        summands = np.copy(summands)
        active_flags = np.ones(len(summands), dtype=np.bool)

        for num, _ in enumerate(summands):
            if active_flags[num]:
                other_gr = np.where((groups != groups[num]))[0] # groups diff from current target
                other_gr = other_gr[other_gr > num]
                for num2 in other_gr:
                    if np.all(np.in1d(summands[num2][summands[num2] > 0], summands[num])):
                        active_flags[num2] = False
                        groups[num2] = groups[num]

        # Choose union

        active_indexs = np.arange(active_flags.shape[0])[active_flags == True]

        for ind_main in active_indexs:
            if ind_main in np.arange(active_flags.shape[0])[active_flags == True]:
                merge_score = np.zeros(len(summands)) - 1
                main = np.unique(summands[ind_main])
                if len(main[main > 0]) >= T:
                    continue
                for ind_part in np.arange(active_flags.shape[0])[active_flags == True]:
                    if ind_main < ind_part:
                        part = np.unique(summands[ind_part])
                        main = main[main > 0]
                        part = part[part > 0]
                        unite = np.unique(np.union1d(main, part))
                        uniq_len = len(unite[unite > 0])  # free space
                        if uniq_len > T:
                            merge_score[ind_part] = -1
                            continue
                        new_len = len(unite[unite > 0]) - len(main[main > 0])
                        copylen = len(part[part > 0]) - new_len
                        if DEBUG and not new_len:
                            raise Exception('zero new len {} {}'.format(summands[ind_main], summands[ind_part]))
                        merge_score[ind_part] = copylen / new_len

                if max(merge_score) >= 0:
                    m_arg = np.argmax(merge_score)
                    tmp_summand = np.zeros(summands.shape[1], dtype=summands.dtype)
                    tmp_summand_vars = np.unique(np.union1d(summands[ind_main], summands[m_arg]))
                    tmp_summand_vars = tmp_summand_vars[tmp_summand_vars > 0]
                    tmp_summand[:len(tmp_summand_vars)] = tmp_summand_vars
                    summands[ind_main] = tmp_summand
                    groups[groups == groups[m_arg]] = groups[ind_main]

                    active_flags[m_arg] = False

                    # Merge new
                    for num in (np.where([groups != groups[ind_main]][0] & active_flags))[0]:
                        if np.all(np.in1d(summands[num][summands[num] > 0], summands[ind_main])):
                            active_flags[num] = False
                            groups[num] = groups[ind_main]

        # getgroups
        _, inverse = np.unique(groups, return_inverse=True)
        groups = inverse
        result = dict()
        for i in groups:
            result[i] = np.unique(summands[groups == i])
        return result, groups


    def small_poly_to_cnf(self, poly, const=False):
        """
        Берет полином длинной меньше Т, и превращает его в КНФ методом Куайна — Мак-Класки (#TODO kmap)
        :param poly: ZhegalkinPoly
        :param const: const True/False, constat monom
        :return: stat [len, deg, rg]
        """

        # 1 poly to pknf

        table = self.TruthTable(poly)
        p_cnf = np.where(~table.tt_right)[0]

        # 2 sknf to min knf
        # Quine–McCluskey algorithm

        def count_one(x):
            """
               Считает единицы в бинарном представлении х
            """
            x = (x & 0x55555555) + ((x >> 1) & 0x55555555)
            x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
            x = (x & 0x0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f)
            x = (x & 0x00ff00ff) + ((x >> 8) & 0x00ff00ff)
            x = (x & 0x0000ffff) + ((x >> 16) & 0x0000ffff)
            return x

        def make_implicants(old_groups):
            # импликанты для которых нашлись "пары" с 2 вместо -
            new_groups = {num: [] for num in range(max(old_groups.keys()) + 1)}
            end_implicants = []  # тупиковые ипмликанты

            for num in old_groups.keys():
                current_group = old_groups.get(num, 'Empty')
                next_group = old_groups.get(num + 1, 'Empty')
                if current_group != 'Empty' and next_group != 'Empty':
                    for minterm in current_group:
                        possible_pairs = next_group[np.all((next_group == 2) == (minterm == 2), axis=1)]
                        possible_pairs = possible_pairs[np.all((minterm & possible_pairs) == minterm, axis=1)]
                        if len(possible_pairs):
                            for possible_pair in possible_pairs:
                                tmp_copy = np.copy(possible_pair)
                                tmp_copy[(tmp_copy - minterm) == 1] = 2
                                new_groups[num].append(tmp_copy)
                        else:
                            end_implicants.append(minterm)

            return np.array(new_groups), end_implicants

        groups = dict()
        data = sorted(p_cnf, key=count_one)
        for k, g in groupby(data, count_one):
            g = np.array([table.tt_left[c1].astype(np.uint8) for c1 in g])
            groups[k] = g

        print(make_implicants(groups))
        # iter_num = 0
        # while True:
        #     iter_num += 1
        #     groups, implicants = make_implicants(groups)

#
# def generate_summands(num_ran=[2, 10], num_sum=5, num_width=6):
#     vars_names = np.arange(*num_ran)
#     vars_names = np.r_[vars_names, np.zeros_like(vars_names)]
#     res = np.array([np.random.choice(vars_names, num_width, replace=0)])
#     res[0].sort()
#     for _ in range(num_sum):
#         for i in range(1000):
#             applicant = np.array([np.random.choice(vars_names, num_width, replace=0)])
#             applicant.sort()
#             if not np.any(~np.any(res - applicant, axis=1)) and (not np.all(applicant == 0)):
#                 break
#         else:
#             raise Exception('cant make impls')
#         res = np.r_[res, applicant]
#     print(repr(res))

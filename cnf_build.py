from itertools import product, groupby

import mock
import numpy as np

from kuz_poly import ZhegalkinPolynomial

DEBUG = True


class CNF_builder:
    """
    Класс для превращения полиномов в окломинимальные КНФ и получения их статистики
    """

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

            tt_left = np.array(list(product((1, 0), repeat=10)), dtype=np.bool)
            res = []
            for i in tt_left:
                res.append(inp_poly.solve_poly(vars_nums[i]))
            self.tt_right = np.array(res, dtype=np.bool)
            if self.const:
                self.tt_right = ~self.tt_right

        class TrTabException(Exception):
            pass

    @staticmethod
    def grouper(summands, T):
        """
        Группирует суманды
        в группы с числом уникальных перменных меньше либо равных T
        :param summands: Массив чисел
        :param T:
        :return:
        """
        # summands unique
        # sorted !
        # filter too big
        groups = np.arange(len(summands))

        # Merge all
        summands = np.copy(summands)
        active_flags = np.ones(len(summands), dtype=np.bool)

        for num, _ in enumerate(summands):
            if active_flags[num]:
                tmp_gr = np.where((groups != groups[num]))[0]
                tmp_gr = tmp_gr[tmp_gr > num]
                for num2 in tmp_gr:
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
                for ind_part in active_indexs:
                    if ind_main != ind_part:
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
        gr = np.unique(groups)
        a = dict()
        for i in gr:
            a[i] = np.unique(summands[groups == i])
        return groups, a

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


        # In [42]: expr2truthtable((x | y | z) & (~x | ~y | z))
        # Out[42]:
        # z y x
        # 0 0 0 : 0
        # 0 0 1 : 1
        # 0 1 0 : 1
        # 0 1 1 : 0
        # 1 0 0 : 1
        # 1 0 1 : 1
        # 1 1 0 : 1
        # 1 1 1 : 1

        def count_one(x):
            """
               Считает единицы в бинарном представлении х
            """
            x = (x & 0x55555555) + ((x >> 1)  & 0x55555555)
            x = (x & 0x33333333) + ((x >> 2)  & 0x33333333)
            x = (x & 0x0f0f0f0f) + ((x >> 4)  & 0x0f0f0f0f)
            x = (x & 0x00ff00ff) + ((x >> 8)  & 0x00ff00ff)
            x = (x & 0x0000ffff) + ((x >> 16) & 0x0000ffff)
            return x

        def make_imlicants(groups):
            for ones in range(max(groups.keys())):
                current_group = groups.get(ones)
                next_group = groups.get(ones.get(ones + 1))
                if current_group and next_group:
                    pass # Search implicants

        groups = dict()
        data = sorted(p_cnf, key=count_one)
        for k, g in groupby(data, count_one):
            g = [table.tt_left[c1] for c1 in g]
            groups[k] = g
        while True:
            pass


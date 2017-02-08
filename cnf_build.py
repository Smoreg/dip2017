import numpy as np
from kuz_poly import ZhegalkinPolynomial
import mock
DEBUG = True


class CNF:
    pass


def grouper(summands, T):
    # summands unique
    # sorted !
    # filter too big
    T = T
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


def small_poly_to_cnf(poly, const=False):
    """
    Берет полином ращзмером меньше Т, и превращает его в КНФ методом Куайна — Мак-Класки (#TODO kmap)
    :param poly: ZhegalkinPoly
    :param const: const True/False, constat monom
    :return: stat [len, deg, rg]
    """





if __name__ == '__main__':
    s = np.array(
        [
            [3, 4, 5],
            [4, 6, 7],
            [4, 5, 0],
            [2, 5, 0],
            [6, 0, 0],
            [2, 0, 0]
        ])
    s = np.c_[s, np.zeros_like(s), np.zeros_like(s), np.zeros_like(s)]  # fill with zeros

    cipher_mock = mock.MagicMock()
    cipher_mock.th = 5
    cipher_mock.T = 12
    cipher_mock.max_deg = 256

    z = ZhegalkinPolynomial(cipher_mock)
    l,h = s.shape
    z.form[:l,:h] = s
    small_poly_to_cnf(s)
    # small_poly_to_cnf(s, True)
    # s = np.c_[s, np.zeros_like(s), np.zeros_like(s), np.zeros_like(s)]
    # for num in range(3,6):
    #     print(num)
    #     res, a = grouper(s,num)
    #     print(np.c_[s, res])
    #     print(a)

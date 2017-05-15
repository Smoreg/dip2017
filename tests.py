import io
import json
import random

import mock
import numpy as np
import pytest
from pygost.gost3412 import GOST3412Kuz

from ciphers import Kuznechik
from transform.bool_structs import ZhegalkinPolynomial, VarSpace, PolyList, SboxPolyTransform
from transform.cnf_transformations import CNF_builder

th = 3
T = 5


class TestCNF:
    @pytest.fixture
    def cipher(self):
        cipher_mock = mock.MagicMock()
        cipher_mock.th = 5
        cipher_mock.T = 5
        cipher_mock.max_deg = 256
        return cipher_mock

    def test_grouper(self, cipher):
        builder = CNF_builder(cipher)

        def get_grouped(test_summands):
            ts0, ts1 = test_summands.shape
            z = np.zeros((ts0, T - ts1), dtype=np.uint32)
            assert ts1 <= builder.cipher.T
            test_summands = np.c_[test_summands, z]
            return builder._grouper(test_summands)

        # simple
        result, _ = get_grouped(
            np.array([
                [2, 3, 4, ],
                [2, 3, 0, ],
                [6, 0, 0, ],
            ])
        )
        assert len(result) == 1
        assert np.all(result[0] == np.array([0, 2, 3, 4, 6]))

        # real simple

        result, _ = get_grouped(
            np.array([
                [0, 3, 5, 9, 2],
                [0, 5, 6, 0, 0],
                [8, 0, 0, 0, 0],
            ])
        )
        assert len(result) == 2

        big_sum = [[0, 3, 5, 9, 2],
                   [0, 5, 6, 0, 0],
                   [8, 0, 0, 0, 0],
                   [0, 0, 5, 4, 7],
                   [0, 0, 5, 8, 0],
                   [5, 0, 4, 3, 7],
                   [9, 10, 5, 0, 0],
                   [9, 8, 5, 6, 0],
                   [2, 5, 8, 0, 3],
                   [6, 2, 0, 0, 7],
                   [0, 0, 3, 6, 0],
                   [6, 4, 9, 5, 8],
                   [3, 0, 4, 2, 9],
                   [0, 3, 2, 4, 0],
                   [3, 0, 6, 0, 9],
                   [8, 0, 7, 0, 6],
                   [9, 3, 0, 0, 10],
                   [0, 9, 7, 4, 0],
                   [10, 4, 2, 6, 0],
                   [6, 0, 2, 3, 10]
                   ]

        result1, groups1 = get_grouped(
            np.array(big_sum)
        )
        result2, groups2 = get_grouped(
            np.array(big_sum * 7)
        )
        assert len(result1) == len(result2)

        check_dict = {gr_num: [] for gr_num in result2.keys()}
        res_eva = True
        for num, summand in enumerate(big_sum * 7):
            res_eva &= np.all(np.in1d(summand, result2[groups2[num]]))
            check_dict[groups2[num]].extend(list(summand))
        result, gr = get_grouped(
            np.array([
                [10, 3, 5, 9, 2],
            ])
        )
        assert len(result) == 1

        # #3 exceptions (too long, empty, not sorted, etc)
        # TODO

    def test_poly_to_cnf(self, cipher):
        builder = CNF_builder(cipher)
        s = np.array(
            [
                [3, 4, 5],
                [2, 5, 0],
                [3, 5, 0],
                [5, 0, 0],
            ]
        )

        # 2345
        # 0000 - 0
        # 0001 - 1
        # 0010 - 0
        # 0011 - 1
        # 0100 - 0
        # 0101 - 0
        # 0110 - 0
        # 0111 - 1
        # 1000 - 0
        # 1001 - 0
        # 1010 - 0
        # 1011 - 0
        # 1100 - 0
        # 1101 - 1
        # 1110 - 0
        # 1111 - 0
        res = [bool(x == '1') for x in '0101000100000100']

        s = np.c_[s, np.zeros_like(s), np.zeros_like(s), np.zeros_like(s)]  # fill with zeros

        z1 = ZhegalkinPolynomial(cipher)
        l, h = s.shape
        z1.form[:l, :h] = s

        tt_good = builder.TruthTable(z1)

        assert np.all(tt_good.tt_right == res)

        z1.const ^= True
        tt_good_c = builder.TruthTable(z1)

        assert not np.any(tt_good_c.tt_right == tt_good.tt_right)

        s = np.array(
            [
                [0, 0, 0],
            ])
        s = np.c_[s, np.zeros_like(s), np.zeros_like(s), np.zeros_like(s)]  # fill with zeros

        z2 = ZhegalkinPolynomial(cipher)
        l, h = s.shape
        z2.form[:l, :h] = s


        # builder.small_poly_to_cnf(z)
        # builder.small_poly_to_cnf(s)
        # builder.small_poly_to_cnf(s, True)
        # s = np.c_[s, np.zeros_like(s), np.zeros_like(s), np.zeros_like(s)]
        # for num in range(3,6):
        #     print(num)
        #     res, a = grouper(s,num)
        #     print(np.c_[s, res])
        #     print(a)

        # def test_stop(self):
        #     assert (1 == 0)


class TestTrTable:
    @pytest.fixture
    def poly(self):
        cipher_mock = mock.MagicMock()
        cipher_mock.th = 5
        cipher_mock.T = 12
        cipher_mock.max_deg = 256
        poly = ZhegalkinPolynomial(cipher_mock)
        return poly


class TestPolyZhi:
    @pytest.fixture
    def cipher(self):
        open_text = b'Some kuzcip test'
        rand_key = b'\xfb\x04\xf6u?\xd6G\xaa\xe8E\x16\xc9OX\xed@\xeb(L\xf6{\xc3]\xadY\xf9"c\\\x19\x1c\x9f'
        cipher = Kuznechik(T, th, open_text=open_text, key=rand_key, key_exp=False)
        return cipher

    def test_poly(self, cipher):
        th = 3
        T = 5
        test_const_poly = ZhegalkinPolynomial(cipher)
        test_var_poly1 = ZhegalkinPolynomial(cipher)
        test_var_poly2 = ZhegalkinPolynomial(cipher)
        test_var_poly_err = ZhegalkinPolynomial(cipher)
        assert (repr(test_const_poly) == '0')

        # const
        test_const_poly.set_const(1)
        test_var_poly1 ^= test_const_poly
        assert (test_const_poly.const)
        assert (test_const_poly.is_const())
        test_var_poly1 ^= test_const_poly
        assert (test_const_poly.const)

        # vars
        with pytest.raises(ZhegalkinPolynomial.ZhegalkinException):
            test_var_poly_err.add_summands([[2, 3, 4], [5, 6, 7]])
        with pytest.raises(ZhegalkinPolynomial.ZhegalkinException):
            test_var_poly_err.add_summands(np.arange(16, dtype=np.int8).reshape(2, 8) + 3)

        summand1 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand1[0:2] = np.array([2, 3], dtype=np.int32)
        summand2 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand2[0:2] = np.array([3, 4], dtype=np.int32)

        test_var_poly1.add_summands(np.array([summand1, summand2]))
        test_var_poly1.add_summands(np.array([summand1 + 2, ]))

        assert (not test_var_poly1.is_const())
        assert (not (test_var_poly1 ^ test_const_poly).is_const())
        assert (not any(test_const_poly.form.ravel()))  # tets

        tmp = test_var_poly1 ^ test_var_poly2
        test_var_poly1 ^= test_var_poly2

        assert (tmp == test_var_poly1)

    def test_poly_xor(self, cipher):
        th = 3
        T = 5

        test_const_poly = ZhegalkinPolynomial(cipher)
        test_var_poly1 = ZhegalkinPolynomial(cipher)
        test_var_poly2 = ZhegalkinPolynomial(cipher)
        test_var_poly3 = ZhegalkinPolynomial(cipher)

        test_const_poly.set_const(1)

        summand1 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand1[0:2] = np.array([2, 3], dtype=np.int32)
        summand2 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand2[0:2] = np.array([3, 4], dtype=np.int32)
        summand3 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand3[0:3] = np.array([2, 3, 4], dtype=np.int32)

        test_var_poly1.add_summands(summand1.reshape(1, cipher.max_deg))
        test_var_poly2.add_summands(summand2.reshape(1, cipher.max_deg))
        test_var_poly3.add_summands(summand3.reshape(1, cipher.max_deg))

        test_var_poly_res = test_var_poly1 ^ test_var_poly2 ^ test_var_poly3
        assert np.all(test_var_poly_res.form[0] == summand1)
        assert np.all(test_var_poly_res.form[1] == summand2)
        assert np.all(test_var_poly_res.form[2] == summand3)

        test_var_poly_res ^= test_var_poly1 ^ test_var_poly2 ^ test_var_poly3

        assert (
            np.all(
                test_var_poly_res.form == np.zeros_like(test_var_poly_res.form)
            )
        )

    def test_poly_solve(self, cipher):
        cipher.th = 5
        T = 5

        test_var_poly1 = ZhegalkinPolynomial(cipher)
        test_var_poly2 = ZhegalkinPolynomial(cipher)
        test_var_poly3 = ZhegalkinPolynomial(cipher)

        # test_const_poly.set_const(1)

        summand1 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand1[0:2] = np.array([2, 10], dtype=np.int32)
        summand2 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand2[0:2] = np.array([2, 3], dtype=np.int32)
        summand3 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand3[0:2] = np.array([20, 30], dtype=np.int32)

        test_var_poly1.add_summands(summand1.reshape(1, cipher.max_deg))
        test_var_poly2.add_summands(summand2.reshape(1, cipher.max_deg))
        test_var_poly3.add_summands(summand3.reshape(1, cipher.max_deg))

        res1 = test_var_poly1.solve_poly(true_variables=np.array([2, ]))
        test_var_poly1.set_const(True)
        assert ((not res1) == test_var_poly1.solve_poly(true_variables=np.array([2, ])))
        assert (test_var_poly2.solve_poly(true_variables=np.array([2, 3])))
        assert (not test_var_poly2.solve_poly(true_variables=np.array([3, ]), false_variables=np.array([2, ])))
        assert (test_var_poly3.solve_poly(true_variables=np.array([20, 5, 30])))
        # assert (not (test_var_poly3 ^ test_var_poly2).solve_poly(true_variables=np.array([2, 3, 5, 20, 30])))

        test_var_poly2.set_const(True)
        assert (not test_var_poly2.solve_poly(true_variables=np.array([2, 3])))


class TestVarSpace:
    # Examples
    # True  vars [  1   3   6   7   9  10  12  13  14  15  17  18  20  21  23  25  26
    # False vars [  0,  2,  4,  5,  8, 11, 16, 19, 22, 24, 27, 28, 30, 32, 33, 35, 36,

    @pytest.fixture
    def cipher(self):
        th = 4
        T = 5
        open_text = b'Some kuzcip test'
        rand_key = b'\xfb\x04\xf6u?\xd6G\xaa\xe8E\x16\xc9OX\xed@\xeb(L\xf6{\xc3]\xadY\xf9"c\\\x19\x1c\x9f'
        cipher = Kuznechik(T, th, open_text=open_text, key=rand_key, key_exp=False)
        return cipher

    @pytest.fixture
    def vs(self, cipher):
        original_plaintext = cipher.original_plaintext
        original_key = cipher.original_key

        return VarSpace(
            variables=np.hstack((
                original_plaintext,
                original_key
            )),
            cipher=cipher,
            save_flag=True
        )

    def test_init(self, vs, cipher):
        var_space = vs

        assert np.all(var_space.variables[:128] == cipher.original_plaintext)
        assert isinstance(var_space.sf, io.IOBase)

    def test_xor(self, vs, cipher):
        xor_poly = ZhegalkinPolynomial(cipher)
        xor_poly.form[:7, :5] = np.array(
            [[0, 0, 3, 4, 5],
             [0, 0, 0, 2, 8],
             [0, 0, 3, 4, 8],
             [0, 0, 4, 5, 6],
             [0, 3, 4, 5, 9],
             [0, 0, 2, 4, 8],
             [0, 0, 2, 3, 9],
             ])

        vs.xor(xor_poly.form, const=False)

        # true_vars = np.where(cipher.original_key == 1)[0] + 2
        # solve = xor_poly.solve_poly(true_vars)


#
# groups = vs.cnf_builder._grouper(summands=xor_poly.form)
#
# group1 = [groups[i] for i in group1]
# group2 = [groups[i] for i in group2]
#
# assert group1[1:] == group1[:-1]
# assert group2[1:] == group2[:-1]
# assert min(groups) >= 0
#
# xor_res, stat = vs.new_var(xor_poly)
#
# assert xor_res.form[0, 0] != vs.var_stat['nvar'] - 1
# assert np.all(xor_res.form.ravel()[1:] == 0)
# assert vs.variables[vs.var_stat['nvar'] - 1] == solve
#
# # With long
# xor_poly = ZhegalkinPolynomial(cipher)
# xor_poly.form[0, :8] = np.array(
#     [2, 3, 4, 5, 6, 7, 8, 9])
# #       [4, 5, 0],
# #       [2, 5, 0],
# #       [6, 0, 0],
# #       [4, 6, 7],
# #       [2, 0, 0]
#
# true_vars = np.where(cipher.original_key == 1) + 2
# solve = xor_poly.solve_poly(true_vars)
#
# group1 = [1, 2, 5]
# group2 = [3, 4]
# group_min = [0]
#
#
# group_min = [groups[i] for i in group_min]
# assert max(group_min) < 0
# assert group_min[1:] == group_min[:-1]
#
# xor_res, stat = vs.new_var(xor_poly)
#
# assert xor_res.form[0, 0] != vs.var_stat['nvar'] - 1
# assert np.all(xor_res.form.ravel()[1:] == 0)
# assert vs.variables[vs.var_stat['nvar'] - 1] == solve
#
#
# # assert stat == [len, deg, rg]


class TestPolyList:
    @pytest.fixture
    def cipher(self):
        open_text = b'Some kuzcip test'
        rand_key = b'\xfb\x04\xf6u?\xd6G\xaa\xe8E\x16\xc9OX\xed@\xeb(L\xf6{\xc3]\xadY\xf9"c\\\x19\x1c\x9f'
        cipher = Kuznechik(T, th, open_text=open_text, key=rand_key, key_exp=False)
        return cipher

    def test_polylist(self, cipher):
        cipher.th = 5
        vars_r1 = np.random.randint(2, size=(126 + 258))
        vars_r2 = np.random.randint(2, size=(126 + 258))
        vars_not_const = np.array(
            [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        vars_z = np.zeros_like(vars_r1)

        test_list_r1 = PolyList(vars_r1, 4, cipher)
        test_list_r2 = PolyList(vars_r2, 4, cipher)
        test_list_z = PolyList(vars_z, 4, cipher)
        test_list_not_const = PolyList(vars_not_const, 4, cipher)

        assert (test_list_r1.is_const())
        assert (test_list_r2.is_const())
        assert (test_list_z.is_const())
        assert (not test_list_not_const.is_const())

        vars_numbs = vars_not_const[vars_not_const > 1]

        assert np.all(test_list_r1.solve_list() == vars_r1)
        assert np.all(test_list_not_const.solve_list(true_variables=vars_numbs)[vars_not_const > 1])
        assert ~np.any(test_list_not_const.solve_list(true_variables=vars_numbs)[vars_not_const <= 1])

    def test_polylist_xor(self, cipher):
        na = Kuznechik._narr_to_by_arr
        cipher.th = 5
        T = 5

        test_var_poly1 = ZhegalkinPolynomial(cipher)
        test_var_poly2 = ZhegalkinPolynomial(cipher)
        test_var_poly3 = ZhegalkinPolynomial(cipher)

        # test_const_poly.set_const(1)



        summand1 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand1[0:2] = np.array([2, 10], dtype=np.int32)
        summand2 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand2[0:2] = np.array([2, 3], dtype=np.int32)
        summand3 = np.zeros(cipher.max_deg, dtype=np.int32)
        summand3[0:2] = np.array([4, 5], dtype=np.int32)

        test_var_poly1.add_summands(summand1.reshape(1, cipher.max_deg))
        test_var_poly2.add_summands(summand2.reshape(1, cipher.max_deg))
        test_var_poly3.add_summands(summand3.reshape(1, cipher.max_deg))

        test_x0 = bytearray(b'\xfb\x04\xf6u?\xd6G\xaa\xe8E\x16\xc9OX\xed@')
        test_x1 = bytearray(b"n\xa2vrlHz\xb8]\'\xbd\x10\xdd\x84\x94\x01")
        test_x2 = bytearray(b"\x00" * 16)

        def strxor(a, b):
            """ pygostxor"""
            mlen = min(len(a), len(b))
            a, b, xor = bytearray(a), bytearray(b), bytearray(mlen)
            for i in range(mlen):
                xor[i] = a[i] ^ b[i]
            return bytes(xor)

        poly0 = PolyList(variables=test_x0, th=th, cipher=cipher)
        assert (bytearray(strxor(test_x0, test_x1))) == (na((poly0 ^ test_x1).solve_list()))
        assert (bytearray(strxor(test_x0, test_x1))) == (na((poly0 ^ test_x1).solve_list()))
        assert (bytearray(strxor(test_x0, test_x2))) == (na((poly0 ^ test_x2).solve_list()))
        assert (bytearray(strxor(test_x0, test_x2))) == (na((poly0 ^ test_x2).solve_list()))


class TestKuznechik:
    T = 4
    th = 4

    open_text = b'Some kuzcip test'
    rand_key = b'\xfb\x04\xf6u?\xd6G\xaa\xe8E\x16\xc9OX\xed@\xeb(L\xf6{\xc3]\xadY\xf9"c\\\x19\x1c\x9f'

    pygost_kuz = GOST3412Kuz(rand_key)
    cipher_text = pygost_kuz.encrypt(open_text)

    def test_key_expand_const(self):
        test_kuz = Kuznechik(self.T, self.th, open_text=self.open_text, key=self.rand_key)

        assert len(test_kuz.full_key) == len(self.pygost_kuz.ks)

        for num, ij in enumerate(zip(test_kuz.full_key, self.pygost_kuz.ks)):
            i, j = ij
            assert test_kuz._narr_to_by_arr(i) == j, "Round key {} fail".format(num)

        assert (test_kuz._narr_to_by_arr(test_kuz.encrypt()) == bytearray(self.pygost_kuz.encrypt(self.open_text)))


# open_text = b'Some kuzcip test'
# rand_key = b'\xfb\x04\xf6u?\xd6G\xaa\xe8E\x16\xc9OX\xed@\xeb(L\xf6{\xc3]\xadY\xf9"c\\\x19\x1c\x9f'
# cipher = Kuznechik(T, th, open_text=open_text, key=rand_key, key_exp=False)

# open_text = b'Some kuzcip test'
# rand_key = b'\xfb\x04\xf6u?\xd6G\xaa\xe8E\x16\xc9OX\xed@\xeb(L\xf6{\xc3]\xadY\xf9"c\\\x19\x1c\x9f'
# cipher = 1

class TestSbox:
    def test_sbox_main(self):
        PI = bytearray((
            252, 238, 221, 17, 207, 110, 49, 22, 251, 196, 250, 218, 35, 197, 4, 77,
            233, 119, 240, 219, 147, 46, 153, 186, 23, 54, 241, 187, 20, 205, 95, 193,
            249, 24, 101, 90, 226, 92, 239, 33, 129, 28, 60, 66, 139, 1, 142, 79, 5,
            132, 2, 174, 227, 106, 143, 160, 6, 11, 237, 152, 127, 212, 211, 31, 235,
            52, 44, 81, 234, 200, 72, 171, 242, 42, 104, 162, 253, 58, 206, 204, 181,
            112, 14, 86, 8, 12, 118, 18, 191, 114, 19, 71, 156, 183, 93, 135, 21, 161,
            150, 41, 16, 123, 154, 199, 243, 145, 120, 111, 157, 158, 178, 177, 50, 117,
            25, 61, 255, 53, 138, 126, 109, 84, 198, 128, 195, 189, 13, 87, 223, 245,
            36, 169, 62, 168, 67, 201, 215, 121, 214, 246, 124, 34, 185, 3, 224, 15,
            236, 222, 122, 148, 176, 188, 220, 232, 40, 80, 78, 51, 10, 74, 167, 151,
            96, 115, 30, 0, 98, 68, 26, 184, 56, 130, 100, 159, 38, 65, 173, 69, 70,
            146, 39, 94, 85, 47, 140, 163, 165, 125, 105, 213, 149, 59, 7, 88, 179, 64,
            134, 172, 29, 247, 48, 55, 107, 228, 136, 217, 231, 137, 225, 27, 131, 73,
            76, 63, 248, 254, 141, 83, 170, 144, 202, 216, 133, 97, 32, 113, 103, 164,
            45, 43, 9, 91, 203, 155, 37, 208, 190, 229, 108, 82, 89, 166, 116, 210, 230,
            244, 180, 192, 209, 102, 175, 194, 57, 75, 99, 182,
        ))
        test_instance = SboxPolyTransform(sbox=PI)
        assert test_instance

    def test_sbox_rand(self):
        rand_sbox = bytearray((
            3, 1, 2, 0
        ))

        test_instance2 = SboxPolyTransform(sbox=rand_sbox)
        assert np.all(
            test_instance2.general_polys[0].form == \
            [np.array([1, 0], dtype=np.int32),
             np.array([0, 0], dtype=np.int32),
             np.array([0, 0], dtype=np.int32),
             np.array([0, 0], dtype=np.int32)]
        )
        assert np.all(
            test_instance2.general_polys[1].form == \
            [np.array([2, 0], dtype=np.int32),
             np.array([0, 0], dtype=np.int32),
             np.array([0, 0], dtype=np.int32),
             np.array([0, 0], dtype=np.int32)])

    def test_sbox_true_rand(self):
        # sbox_size = random.randint(2, 8)
        sbox_size = 3
        sbox = bytearray(list(range(2 ** sbox_size)))
        random.shuffle(sbox)
        test_instance3 = SboxPolyTransform(sbox)
        assert test_instance3

        for num, bit_vec in enumerate(sbox):
            bit_vec_norm = [int(x) for x in list(bin(num)[2:].rjust(sbox_size, '0'))]
            bit_res_norm = [int(x) for x in list(bin(bit_vec)[2:].rjust(sbox_size, '0'))]
            assert bit_res_norm == test_instance3.vector_solve(bit_vec_norm)


        res = [[] for _ in range(sbox_size)]
        for c1 in range(2 ** sbox_size):
            # TODO нормальный привод в битовому списку
            bit_vector = np.array([int(x) for x in list(bin(c1)[2:].rjust(sbox_size, '0'))])
            for num, poly in enumerate(test_instance3.general_polys):
                res[num].append(
                    int(
                        poly.solve_poly(true_variables=np.where(bit_vector == 1)[0])
                    )
                )
            # res = [''.join(x) for x in res]
            print(json.dumps(res))
        a = np.array(res)

        print(123)
            # x2 + 1
            # 1 + x1 + x1x2

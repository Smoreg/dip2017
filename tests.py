import io

import mock
import numpy as np
import pytest
from pygost.gost3412 import GOST3412Kuz

from cnf_build import CNF_builder
from kuz_poly import ZhegalkinPolynomial, Kuznechik, VarSpace, PolyList

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
            return builder.grouper(test_summands)

        # simple
        result = get_grouped(
            np.array([
                [2, 3, 4, ],
                [2, 3, 0, ],
                [6, 0, 0, ],
            ])
        )
        assert len(result) == 1
        assert np.all(result[0] == np.array([0, 2, 3, 4, 6]))

        # real

        result = get_grouped(
            np.array([
                [0, 3, 5, 9, 2],
                [0, 5, 6, 0, 0],
                [8, 0, 0, 0, 0],
            ])
        )
        assert len(result) == 1
        assert np.all(result[0] == np.array([0, 2, 3, 4, 6]))


        # #2 no diff
        # test_summands = np.array([
        #     [2, 3, 4, ],
        # ])
        # ts0, ts1 = test_summands.shape
        # assert ts1 <= builder.cipher.T
        #
        # z = np.zeros((ts0, T - ts1), dtype=np.uint32)
        # test_summands = np.c_[test_summands, z]
        # result = builder.grouper(test_summands)
        # assert result != test_summands
        # #3 exceptions (too long, empty, not sorted, etc)

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
            th=cipher.th,
            T=cipher.T,
            save_flag=True
        )

    def test_init(self, vs, cipher):
        var_space = vs

        assert var_space.variables[(np.where(cipher.original_key == 1) + 2)[5]]
        assert not var_space.variables[(np.where(cipher.original_key == 0) + 2)[5]]
        assert isinstance(var_space.sf, io.IOBase)

    def test_xor(self, vs, cipher):
        xor_poly = ZhegalkinPolynomial(cipher)
        xor_poly.form[:6, :3] = np.array(
            [
                [3, 4, 5],
                [4, 5, 0],
                [2, 5, 0],
                [6, 0, 0],
                [4, 6, 7],
                [2, 0, 0]
            ])
        true_vars = np.where(cipher.original_key == 1) + 2
        solve = xor_poly.solve_poly(true_vars)

        group1 = [0, 1, 2, 5]
        group2 = [3, 4]

        groups = vs.group_monoms(summands=xor_poly.form)

        group1 = [groups[i] for i in group1]
        group2 = [groups[i] for i in group2]

        assert group1[1:] == group1[:-1]
        assert group2[1:] == group2[:-1]
        assert min(groups) >= 0

        xor_res, stat = vs.new_var(xor_poly)

        assert xor_res.form[0, 0] != vs.var_stat['nvar'] - 1
        assert np.all(xor_res.form.ravel()[1:] == 0)
        assert vs.variables[vs.var_stat['nvar'] - 1] == solve

        # With long
        xor_poly = ZhegalkinPolynomial(cipher)
        xor_poly.form[0, :8] = np.array(
            [2, 3, 4, 5, 6, 7, 8, 9])
        #       [4, 5, 0],
        #       [2, 5, 0],
        #       [6, 0, 0],
        #       [4, 6, 7],
        #       [2, 0, 0]

        true_vars = np.where(cipher.original_key == 1) + 2
        solve = xor_poly.solve_poly(true_vars)

        group1 = [1, 2, 5]
        group2 = [3, 4]
        group_min = [0]

        groups = vs.group_monoms(summands=xor_poly.form)

        group_min = [groups[i] for i in group_min]
        assert max(group_min) < 0
        assert group_min[1:] == group_min[:-1]

        xor_res, stat = vs.new_var(xor_poly)

        assert xor_res.form[0, 0] != vs.var_stat['nvar'] - 1
        assert np.all(xor_res.form.ravel()[1:] == 0)
        assert vs.variables[vs.var_stat['nvar'] - 1] == solve


        # assert stat == [len, deg, rg]


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
        na = Kuznechik.narr_to_by_arr
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
            assert test_kuz.narr_to_by_arr(i) == j, "Round key {} fail".format(num)

        assert (test_kuz.narr_to_by_arr(test_kuz.encrypt()) == bytearray(self.pygost_kuz.encrypt(self.open_text)))

# open_text = b'Some kuzcip test'
# rand_key = b'\xfb\x04\xf6u?\xd6G\xaa\xe8E\x16\xc9OX\xed@\xeb(L\xf6{\xc3]\xadY\xf9"c\\\x19\x1c\x9f'
# cipher = Kuznechik(T, th, open_text=open_text, key=rand_key, key_exp=False)

# open_text = b'Some kuzcip test'
# rand_key = b'\xfb\x04\xf6u?\xd6G\xaa\xe8E\x16\xc9OX\xed@\xeb(L\xf6{\xc3]\xadY\xf9"c\\\x19\x1c\x9f'
# cipher = 1

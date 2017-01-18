from copy import deepcopy as copy
import numpy as np
import math
from time import gmtime, strftime

DEBUG = True


class ZhegalkinPolynomial:
    # Legend
    # th = 8
    # [1, 0, 0, 0, 0, 1, 0, 0],
    # [1, 1, 0, 1, 0, 0, 0, 1],
    # [0, 0, 0, 1, 1, 0, 1, 0],
    # [1, 1, 1, 1, 1, 1, 1, 0],
    # [0, 0, 1, 1, 1, 1, 1, 0],
    # [1, 1, 1, 1, 0, 1, 1, 0],
    # [0, 0, 1, 1, 0, 1, 1, 1],
    # [1, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 1, 1, 1, 0, 1])
    # ....

    # 'x1 ⊕ x6',
    # 'x1 ⊕ x2 ⊕ x4 ⊕ x8',
    # 'x4 ⊕ x5 ⊕ x7',
    # 'x1 ⊕ x2 ⊕ x3 ⊕ x4 ⊕ x5 ⊕ x6 ⊕ x7',
    # 'x3 ⊕ x4 ⊕ x5 ⊕ x6 ⊕ x7',
    # 'x1 ⊕ x2 ⊕ x3 ⊕ x4 ⊕ x6 ⊕ x7',
    # 'x3 ⊕ x4 ⊕ x6 ⊕ x7 ⊕ x8',
    # 'x1',
    # 'x3 ⊕ x4 ⊕ x5 ⊕ x6 ⊕ x8']
    # ...
    # ⊕ self.const

    def __init__(self, cipher):
        self.main_cipher = cipher

        self.th = cipher.th
        self.h = 2 ** cipher.th
        self.w = cipher.th

        self.form = np.zeros((self.h, self.w), dtype=np.int32)
        self.const = False

    def __repr__(self):
        if self.get_vars():
            return "{} ^ {}".format(int(self.const), "some vars")
        else:
            return str(int(self.const))

    def __eq__(self, other):
        return np.all(self.form == other.form) and (self.const == other.const)

    def __ixor__(self, other):
        # в любом случае константы xor ятся
        self_c = self.is_const()
        oth_c = other.is_const()
        if self_c or oth_c:
            # если один полином - константа
            if self_c and not (oth_c):
                self.form = np.copy(other.form)
            self.const ^= other.const

        else:

            if self.is_need_new(other, self.th):
                print('MORE')
                # новые переменные #TODO
                # TODO1

            else:
                self.form = self.xor_summands(np.vstack((other.get_summands(), self.get_summands())))
                self.const ^= other.const
        return self

    def __xor__(self, other):
        res = copy(self)
        res ^= other
        return res

    def __deepcopy__(self, *args, **kwargs):
        my_copy = ZhegalkinPolynomial(self.main_cipher)
        my_copy.const = self.const
        my_copy.form = np.copy(self.form)
        return my_copy

    def xor_summands(self, summands):

        sorted_idx = np.lexsort(summands.T)
        summands = summands[sorted_idx, :]

        res = []

        step_over_flag = False

        for num, s in enumerate(zip(summands, summands[1:])):
            if step_over_flag:
                step_over_flag = False
                continue
            s1, s2 = s
            if ~np.any(s1 ^ s2):
                step_over_flag = True
                continue
            else:
                res.append(num)

        if not step_over_flag:
            res.append(len(summands) - 1)
        if res:
            new_summands = summands[np.array(res)]
        else:
            new_summands = []

        res = np.zeros_like(self.form)
        if len(new_summands):
            res[:len(new_summands)] = new_summands
        return res

    def get_vars(self):
        """
        возвращает список уникальных переменных
        :return: int >= 0
        """
        uniq_vars = np.unique(self.form.ravel())
        return uniq_vars[uniq_vars >= 2]

    def get_vars_num(self):
        return len(self.get_vars())

    def is_need_new(self, other, th):
        return len(np.unique(np.append(self.get_vars(), other.get_vars()))) > th

    def new_vars(self, variables):
        """
        возращает переменные которые нужно добавить в список переменных
        :param variables:
        :return:
        """
        return np.setdiff1d(
            variables,
            self.get_vars(),
            assume_unique=True)

    def set_const(self, const):
        self.const = bool(const)

    def get_empty_summands_room_indexes(self):
        return np.where(np.all(self.form == np.zeros(self.form.shape[1], dtype=np.bool), axis=1))[0]

    def get_busy_summands_room_indexes(self):
        return np.where(np.any(self.form != np.zeros(self.form.shape[1], dtype=np.bool), axis=1))[0]

    def get_summands(self):
        return self.form[:self.get_empty_summands_room_indexes()[0]]

    def add_summands(self, summands):
        """
        Добавлет слагаемые в полином Жегалкина
        :param summands: Массив переменных слагаемых shape == (число новых слагаемых, максимальная длина монома)
        :type summands: np.ndarray dtype = int32
        :return:
        """
        empty_summands_rooms = self.get_empty_summands_room_indexes()
        if DEBUG and len(empty_summands_rooms) < len(summands):
            raise self.ZhegalkinException('No room for new summand')
        if DEBUG and (not isinstance(summands, np.ndarray)):
            raise self.ZhegalkinException('summands not ndarray')
        if DEBUG and summands.dtype != np.int32:
            raise self.ZhegalkinException('summands bad type {}'.format(summands.dtype))
        else:
            if len(summands.shape) == 1:
                self.form[empty_summands_rooms[0]] = summands
            else:
                self.form[empty_summands_rooms[0]:empty_summands_rooms[0] + len(summands)] = summands

    def is_const(self):
        return ~np.any(self.form)

    def solve_poly(self, true_variables=np.array([], dtype=np.uint32), false_variables=np.array([], dtype=np.uint32)):
        """
        :param true_variables: номера переменных равных 1
        :param false_variables: -//- 0 необязательный
        :return:
        """
        vars_poly = self.get_vars()
        if not any(vars_poly):
            return self.const
        else:
            if DEBUG and not isinstance(true_variables, np.ndarray):
                raise self.ZhegalkinException('vars not ndarray')
            if DEBUG:
                keys = np.hstack([true_variables, false_variables])
                diff = np.setdiff1d(vars_poly, keys, assume_unique=True)
                if len(diff):
                    raise self.ZhegalkinException('Unknown var(s) []'.format(diff))
            summands = self.get_summands()
            bool_vars = \
                (np.in1d(summands, np.append(0, true_variables)).reshape(*summands.shape))
            bool_vars = np.logical_and.reduce(bool_vars, axis=1)
            return np.logical_xor.reduce(bool_vars, axis=0) ^ self.const

    class ZhegalkinException(Exception):
        pass


class PolyList:
    def __init__(self, variables, th, cipher):
        """
        Лист полиномов Жегалкина
        Используется вместо строчек битов обычными алгоритмами шифрования
        :param variables: лист интов [0,1] - константы. Любое другое число - номер перменной
        :param th: >=2 параметр интенсивности введения новых переменных.
        От него так же зависит память отводимая на один полином Жегалкина
        """
        self.th = th
        self.main_cipher = cipher
        if isinstance(variables, bytearray):
            variables = np.unpackbits(variables).astype(np.uint32)
        self.variables = [ZhegalkinPolynomial(self.main_cipher) for _ in variables]
        for number, variable in enumerate(variables):
            if variable in [0, 1]:
                self.variables[number].set_const(variable)
            else:
                self.variables[number].add_summands(np.array([[variable, ], ], dtype=np.int32))

    def __deepcopy__(self, *args, **kwargs):
        new_variables = list()
        for var in self.variables:
            new_variables.append(copy(var))
        my_copy = PolyList([], self.th, self.main_cipher)
        my_copy.variables = new_variables
        return my_copy

    def get_byte(self, num=-1):
        """
        Вовзращает PolyList из 8 перменных self
        :param num: номер байта.
        :return: PolyList
        """
        pass  # TODO 1

    def __repr__(self):
        cur_repr = "PL {} ".format(len(self))
        cur_repr += ''.join([repr(i) for i in self.variables])
        return cur_repr

    def solve_list(self, true_variables=np.array([], dtype=np.int32), false_variables=np.array([], dtype=np.int32)):
        """
        :param true_variables: номера переменных равных 1
        :param false_variables: -//- 0 необязательный
        :return:
        """

        return (np.array([polynome.solve_poly(
            true_variables=true_variables, false_variables=false_variables) for polynome in self.variables]))

    def __getitem__(self, item):
        return self.variables[item]

    def __len__(self):
        return len(self.variables)

    def __xor__(self, other):
        res = copy(self)
        if isinstance(other, bytearray):  # xor with const bytearray
            other = np.unpackbits(other).astype(dtype=np.bool)
            if DEBUG and len(res) != len(other):
                raise res.PolyError('xor lengths not equal')
            for c1, c2 in zip(res.variables, other):
                c1.const ^= c2
            return res
        elif isinstance(other, PolyList):
            if DEBUG and len(res) != len(other):
                raise res.PolyError('xor lengths not equal')
            for num, zhi_poly in enumerate(other):
                res.variables[num] ^= zhi_poly
            return res

    def is_const(self):
        return all([variable.is_const() for variable in self.variables])

    class PolyError(Exception):
        pass


class VarSpace:
    """
    создан для хранения всех переменных, как основных так и дополнительных.
    Создается в Kuznechik, далее на него ссыляются из всех классов
    :param sf: файл в которые будут сохраняться данные
    :param cipher: ссылка на текущий шифрующий класс
    :param curr_var: номер для следующей новой перменной
    :param variables: list переменных
    :param cacl_on_run: рассчет доп переменных на ходу не храня их формулы. Только значения и необходимые стат параметры

    структура перемнных

    1) variables np.array uint32 [... , 2]
       номер/значение/длина/ранг
       [ 2,  1, 1, 1],
       ...
       [20,  1, 33, 22],
    """

    def __init__(self, variables, cipher, save_flag=False, cacl_on_run=False):

        if save_flag:
            # safe file path
            self.sf = open("VarSpace{}_{}".format(
                hex(id(self))[2:], strftime("%Y%m%d_%H%M", gmtime())
            ))
        else:
            self.sf = save_flag

        self.cacl_on_run = cacl_on_run

        self.variables = self._init_vars(variables)
        self.cipher = cipher
        self.curr_var = np.max(self.variables[:, 0])

    # def __ixor__(self, other):
    #     # в любом случае константы xor ятся
    #     self_c = self.is_const()
    #     oth_c = other.is_const()
    #     if self_c or oth_c:
    #         # если один полином - константа
    #         if self_c and not (oth_c):
    #             self.form = np.copy(other.form)
    #         self.const ^= other.const
    #
    #     else:
    #
    #         if self.is_need_new(other, self.th):
    #             print('MORE')
    #             # новые переменные #TODO
    #             # TODO1
    #
    #         else:
    #             self.form = self.xor_summands(np.vstack((other.get_summands(), self.get_summands())))
    #             self.const ^= other.const
    #     return self
c

    def _init_vars(self, variables):
        res = np.zeros([*variables.shape, 4], dtype=np.uint32)
        res[...:0] = np.arange(2 + variables.shape[0])
        res[...:1] = variables
        res[...:2] += 1
        res[...:3] += 1
        return res

    def make_new_var(self, variables, op='XOR'):
        """
        VarSpace получает выражение которое нужно превратить в новую переменную

        :param variables: Выражение, которое нужно превратить в переменную. Может быть двух видов
         1. ZhegalkinPolynomial - для унарных операций
         2. lsit[ZhegalkinPolynomial] - для бинарный операций.
        :param op: операция
         1. Унарные
            -
         2. Бинарные
            2.1 XOR - сложение по модулю 2. 2 полнома
            2.2 SBOX_{SBOX_NAME} - замена по одному из имеющихся sbox`y. Число полиномов зависит от sbox

        :return: Номер новой переменной
        :rtype: numpy.uint32
        """
        if op == 'XOR':
            pass
        pass  # TODO

    def xor_var(self, poly1, poly2):

        if DEBUG and (poly1.is_const() or poly2.is_const()):
            raise self.VarSpaceException('Const in var space!')

    def sbox_var(self, poly, sbox_name):
        pass

    def add_new_var(self, var):
        # if DEBUG and()
        pass

    class VarSpaceException(Exception):
        pass


class Kuznechik:
    var_space = None

    def __init__(self, T, th, open_text, key, secret_bits_mask=np.array([False] * (256 + 128)), key_exp=True):
        """

        :param T: int параметр для построения кнф
        :param th: int >=2
        :param open_text: биты. На вход идут все, скрываются позже
        :param key: биты
        :param secret_bits_mask: 128 + 256 битов (блок + ключ). 1 - секретн бит
        """
        self.T = T
        self.th = th
        self.secret_bits_mask = secret_bits_mask

        self.precalc = self._precalculate()

        self.original_plaintext = np.unpackbits(np.fromstring(open_text, dtype=np.uint8)).astype(dtype=np.bool)
        self.original_key = np.unpackbits(np.fromstring(key, dtype=np.uint8)).astype(dtype=np.bool)

        self.plaintext = np.where(secret_bits_mask[:128], np.arange(2, 130), self.original_plaintext)
        self.key = np.where(secret_bits_mask[128:], np.arange(130, 386), self.original_key)

        self.poly_plaintext = PolyList(variables=self.original_plaintext, th=self.th, cipher=self)
        self.poly_key = PolyList(variables=self.original_plaintext, th=self.th, cipher=self)

        self.var_space = VarSpace(
            variables=np.hstack((
                self.poly_plaintext.variables,
                self.poly_key.variables
            ))
        )
        if key_exp:
            self.full_key = self.key_expand()
        else:
            self.full_key = None

    @staticmethod
    def narr_to_by_arr(arr, bin_flag=False):
        if isinstance(arr, PolyList):
            arr = arr.solve_list()
        by = bytearray(np.packbits(arr).tobytes())
        if bin_flag:
            return bin(int(by.hex(), base=16))
        else:
            return by

    def lp(self, poly_list):
        """
        стандартная функция кузнечика
        :return:
        """
        if poly_list.is_const():
            blk = np.packbits(poly_list.solve_list())
            res = np.array(self.L([self.precalc.PI[v] for v in blk]), dtype=np.uint8)
            return PolyList(np.unpackbits(res).astype(np.uint32), self.th, self)
        else:
            pass  # TODO

    def L(self, blk, rounds=16):
        if not isinstance(blk, PolyList):
            for _ in range(rounds):
                t = blk[15]
                for i in range(14, -1, -1):
                    blk[i + 1] = blk[i]
                    t ^= self.precalc.GF[blk[i]][self.precalc.LC[i]]
                blk[0] = t
            return blk
        else:
            pass  # TODO

    def key_expand(self):
        kr0 = PolyList(variables=self.key[:16 * 8], th=self.th, cipher=self)
        kr1 = PolyList(variables=self.key[16 * 8:], th=self.th, cipher=self)
        full_key = [copy(kr0), copy(kr1)]
        for i in range(4):
            for j in range(8):
                k = self.lp(kr0 ^ self.precalc.C[8 * i + j])
                kr0, kr1 = k ^ kr1, kr0
            full_key.append(copy(kr0))
            full_key.append(copy(kr1))
        return full_key

    def encrypt(self):
        cipher_text = copy(self.poly_plaintext)
        with open("encrypt_steps_mykuz", "w") as myfile:  # TODEL
            for i in range(9):
                cipher_text = self.lp(self.full_key[i] ^ cipher_text)
                tmp = repr(cipher_text)[3:]
                myfile.write("{} : {} \n".format(i, tmp))  # TODEL

            cipher_text = self.full_key[9] ^ cipher_text
            tmp = repr(cipher_text)[3:]
            myfile.write("{} : {} \n".format(i, tmp))  # TODEL

            return cipher_text

    def _precalculate(self):
        """
        From PYGOST
        """

        LC = bytearray((
            148, 32, 133, 16, 194, 192, 1, 251, 1, 192, 194, 16, 133, 32, 148, 1,
        ))
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

        ########################################################################
        # Precalculate inverted PI value as a performance optimization.
        # Actually it can be computed only once and saved on the disk.
        ########################################################################
        PIinv = bytearray(256)
        for x in range(256):
            PIinv[PI[x]] = x

        def gf(a, b):
            c = 0
            while b:
                if b & 1:
                    c ^= a
                if a & 0x80:
                    a = (a << 1) ^ 0x1C3
                else:
                    a <<= 1
                b >>= 1
            return c

        ########################################################################
        # Precalculate all possible gf(byte, byte) values as a performance
        # optimization.
        # Actually it can be computed only once and saved on the disk.
        ########################################################################
        GF = [bytearray(256) for _ in range(256)]
        for x in range(256):
            for y in range(256):
                GF[x][y] = gf(x, y)

        def L(blk, rounds=16):
            for _ in range(rounds):
                t = blk[15]
                for i in range(14, -1, -1):
                    blk[i + 1] = blk[i]
                    t ^= GF[blk[i]][LC[i]]
                blk[0] = t
            return blk

        def Linv(blk):
            for _ in range(16):
                t = blk[0]
                for i in range(15):
                    blk[i] = blk[i + 1]
                    t ^= GF[blk[i]][LC[i]]
                blk[15] = t
            return blk

        ########################################################################
        # Precalculate values of the C -- it does not depend on key.
        # Actually it can be computed only once and saved on the disk.
        ########################################################################
        C = []
        for x in range(1, 33):
            y = bytearray(16)
            y[15] = x
            C.append(L(y))

        def lp(blk):
            return L([PI[v] for v in blk])

        class PyGost_Precalculation:
            C = None
            # =None

        precalc = PyGost_Precalculation()
        precalc.C = C
        precalc.GF = GF
        precalc.LC = LC
        precalc.PI = PI
        return precalc

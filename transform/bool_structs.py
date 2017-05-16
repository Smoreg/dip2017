# Модуль содержит основные классы для преобразования полиномов Жегалкина согласно алгоритмам шифрования Кузнечик и AES256

from copy import deepcopy
from math import log2
from time import gmtime, strftime

import numpy as np

from transform.cnf_transformations import CNF_builder

DEBUG = True


class FreeZhegalkinPolynomial:
    """
    Полином Жегалкина не зависящий от конкретного шифра
    
    # Legend
    # [3, 4, 9, ... 0, 0, 0, 0],
    # [2, 3, 5, ..., 0, 0, 0, 0],
    # [6, 7, 0, ..., 0, 0, 0, 0],
    # [0, 0, 0, ..., 0, 0, 0, 0],
    # ...
    # 'x3 x4 x9 ⊕ x2 x3 x5 ⊕ x6 x7 ⊕ self.const

    """

    def __init__(self, form, const):
        self.form = form  # форма аналогичная простому ZhegalkinPolynomial
        self.const = const

    def is_const(self):
        return ~np.any(self.form)

    def __repr__(self):
        if not self.is_const():
            res = list()
            for summand in self.get_summands():
                tmp = ''
                for var in summand:
                    if var:
                        tmp += "x{}".format(var)
                if tmp:
                    res.append(tmp)
            res = " ⊕ ".join(res)
            if self.const:
                res = "{} ⊕ {}".format(res, int(self.const))
            return res
        else:
            return str(int(self.const))

    def __eq__(self, other):
        return (self.const == other.const) and np.all(self.form == other.form)

    def __deepcopy__(self, *args, **kwargs):
        return self.__class__(self.form, self.const)

    def get_vars(self):
        """
        возвращает список уникальных переменных
        :return: int >= 0
        """
        uniq_vars = np.unique(self.form.ravel())
        return uniq_vars[uniq_vars >= 2]

    def get_vars_num(self):
        return len(self.get_vars())

    def set_const(self, const):
        self.const = bool(const)

    def get_empty_summands_room_indexes(self):
        return np.where(np.all(self.form == np.zeros(self.form.shape[1], dtype=np.bool), axis=1))[0]

    def get_busy_summands_room_indexes(self):
        return np.where(np.any(self.form != np.zeros(self.form.shape[1], dtype=np.bool), axis=1))[0]

    def get_summands(self):
        empty_rooms = self.get_empty_summands_room_indexes()
        if len(empty_rooms):
            return self.form[:empty_rooms[0]]
        else:
            return self.form

    def add_summands(self, summands):
        """
        Добавлет слагаемые в полином Жегалкина. Без проверки на уникальность
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

    def solve_poly(self, true_variables, false_variables=None):
        """
        :param true_variables: bit mask true vars
        :param false_variables: bit mask false vars
        :return:
        """
        vars_poly = self.get_vars()
        if not any(vars_poly):
            return self.const
        else:
            if DEBUG:
                if false_variables:
                    keys = np.hstack([true_variables, false_variables])
                    diff = np.setdiff1d(vars_poly, keys, assume_unique=True)
                    if len(diff):
                        raise self.ZhegalkinException('Unknown var(s) []'.format(diff))
            summands = self.get_summands()
            bool_vars = \
                (np.in1d(summands, np.append([0, ], true_variables)).reshape(*summands.shape))
            bool_vars = np.logical_and.reduce(bool_vars, axis=1)
            return np.logical_xor.reduce(bool_vars, axis=0) ^ self.const

    class ZhegalkinException(Exception):
        pass


class ZhegalkinPolynomial(FreeZhegalkinPolynomial):
    def __init__(self, cipher, const=False):
        self.cipher = cipher  # Шифр использующий данный класс
        self.th = cipher.th  # Максимальное число мономов (без константы)
        self.monom_max_deg = cipher.max_deg  # Максимальное число пермеенных в полиноме

        form = np.zeros((self.th * 2, self.monom_max_deg * 2), dtype=np.int32)
        super().__init__(form, const)

    def __ixor__(self, other):
        # в любом случае константы xor ятся
        self_c = self.is_const()
        oth_c = other.is_const()
        result = self
        if self_c or oth_c:
            # если один полином - константа
            if self_c and not oth_c:
                self.form = np.copy(other.form)
            self.const ^= other.const
        else:
            state, tmp_form = self.xor_summands(other)
            if state == 'need new':
                result = self.cipher.var_space.xor(self, tmp_form, self.const ^ other.const)
            elif state == 'keep':
                self.form = tmp_form
                self.const ^= other.const
            else:
                raise self.ZhegalkinException('Bad xor state {}'.format(state))
        return result

    def __xor__(self, other):
        res = deepcopy(self)
        res ^= other
        return res

    def __deepcopy__(self, *args, **kwargs):
        my_copy = self.__class__(self.cipher, self.const)
        my_copy.form = np.copy(self.form)
        return my_copy

    def xor_summands(self, other):

        summands = np.vstack((other.get_summands(), self.get_summands()))
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

        if len(new_summands) > self.th or self.get_vars_num() + other.get_vars_num() > self.monom_max_deg:
            return 'need new', new_summands
        else:
            res = np.zeros_like(self.form)
            if len(new_summands):
                res[:len(new_summands)] = new_summands
            return 'keep', res


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
            new_variables.append(deepcopy(var))
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
        res = deepcopy(self)
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
    Класс в который делает те операции над PolyList в результате которых может появиться новая переменная.
    Хранит данные о переменных, а так же назначает новые.
    """
    MAX_VARS = 100000

    def __init__(self, variables, cipher, max_var_num=386, save_flag=False, stat_flag=False):

        if DEBUG:
            if not isinstance(variables, np.ndarray):
                raise self.VarSpaceException('Bad variables type {}'.format(type(variables)))
            if len(variables) != max_var_num - 2:
                raise self.VarSpaceException('Bad var len {} / {}'.format(len(variables), max_var_num))

        self.cnf_builder = CNF_builder(cipher)
        self.sf = save_flag and self._new_file()
        self.f_stat = stat_flag
        self.th = cipher.th
        self.T = cipher.T
        self.cipher = cipher
        self.variables = self._init_vars(variables)  # self.variables[3] = x3

        self.var_stat = {
            "len": 0,
            "nvar": max_var_num,
            "rg": 0,
        }

    def _new_file(self):
        return open("saved_results/VarSpace_{}".format(strftime("%Y-%m-%d-%H-%M", gmtime())
                                                       ), "w")

    def _init_vars(self, variables):
        res = np.zeros(self.MAX_VARS, dtype=np.bool)
        res[2:len(variables) + 2] = variables
        return res

    def make_new_var(self, poly):
        true_vars = np.where(self.variables[2:self.var_stat['nvar']] == 1)[0] + 2  # names of vars equals 1
        res = poly.solve_poly(self.variables, true_variables=true_vars)
        self.variables[self.var_stat['nvar']] = res
        self.var_stat['nvar'] += 1

    # XOR ---------------------------------
    def xor(self, tmp_form, const):
        """
        В процессе операции создаются дополнительные переменные
        :param tmp_form: форма созданная в результате xor двух полиномов. Обычно должна
         создавать xor_summands
        :param const: Константа в фоме
        :return: статистика
        """
        new_poly = ZhegalkinPolynomial(self.cipher, const=const)
        new_poly.form[:tmp_form.shape[0]] = tmp_form
        cnf_stat = self.cnf_builder.small_poly_to_cnf(new_poly)
        new_var = self.var_stat['nvar']
        self.make_new_var(new_poly)
        res_poly = ZhegalkinPolynomial(self.cipher, const=const)
        res_poly.form[:tmp_form.shape[0]] = tmp_form
        return res_poly, cnf_stat

    # SBOX --------------------------------
    def sbox_poly(self, poly_list):
        pass  # TODO

    class VarSpaceException(Exception):
        pass


class SboxPolyTransform:
    """
    Данный класс позволяет преобразовать один PolyList в другой
    в соответвии с данным при инициализации Sbox. 
    
    :argument general_polys: - list of  FreeZhegalkinPolynomial
    
    """

    def __init__(self, sbox):
        """
        Инициация sbox и создание списка полиномов general_polys для обработки каждого байта
        :param sbox: Полный массив входов-выходов sbox bytearray((0, 3, 1, 2,))
        :type sbox: bytearray 
        """
        tmp_len = log2(len(sbox))
        if not tmp_len.is_integer():
            raise self.SboxException('Bad len')
        else:
            self._len = int(tmp_len)
        if max(sbox) > 2 ** self._len - 1:
            raise self.SboxException('Bad sbox')
        self.sbox = sbox
        self.general_polys = self._get_general(sbox)

    def _get_general(self, sbox):
        """
        Метод для заданной sbox составляет соответсвующий каждому входному биту FreeZhegalkinPolynomial
        :param sbox: Полный массив входов-выходов sbox
        :example bytearray((0, 3, 1, 2,)) 
        :return: list of FreeZhegalkinPolynomial 
        """
        general_polys = []
        truth_table = []

        for n, i in enumerate(sbox):
            a1 = bin(n)[2:].rjust(self._len, '0')
            a2 = bin(i)[2:].rjust(self._len, '0')
            truth_table.append((a1, a2))

        for bit_num in range(self._len):
            tmp_truth_table = [(x[0], int(x[1][bit_num])) for x in truth_table]
            general_polys.append(self._truth_table_to_poly(tmp_truth_table))

        return general_polys

    def _truth_table_to_poly(self, tt):

        tt_groups = {num_ones: [] for num_ones in range(self._len + 1)}
        for i in tt:
            tt_groups[i[0].count("1")].append(i)

        poly_coeffs = dict()
        for current_group_num in sorted(tt_groups.keys()):
            for elem in tt_groups[current_group_num]:
                res = elem[1]
                for num_ones_less in range(0, current_group_num):
                    for inner_elem in tt_groups[num_ones_less]:
                        if int(inner_elem[0], base=2) & int(elem[0], base=2) == int(elem[0], base=2):
                            res ^= poly_coeffs[inner_elem[0]]
                poly_coeffs[elem[0]] = res

        const = poly_coeffs['0' * self._len]
        form = np.zeros((2 ** self._len, self._len), dtype=np.int32)
        res_vestors = []
        for bit_combination in sorted(poly_coeffs.keys())[1:]:
            if poly_coeffs[bit_combination]:
                summand = [num + 1 for num, elem in enumerate(bit_combination.rjust(self._len, '0')) if int(elem)]
                summand.extend([0] * (self._len - len(summand)))
                res_vestors.append(
                    np.array(summand)
                )
        res_matrix = np.vstack(res_vestors)
        form[:res_matrix.shape[0]] = res_matrix
        return FreeZhegalkinPolynomial(form, const)

    def __len__(self):
        return self._len

    def poly_list_transform(self, polylist, varspace):
        """
        Преобразует лист полиномов в другой согласно sbox
        Для новых переменных используется varspace
        :param polylist: экземпляр PolyList который будет преобразован 
        :param varspace: экземпляр VarSpace для создания новых переменных
        :return: экземпляр PolyList после sbox преобразования
        """

        if len(polylist) != len(self):
            raise self.SboxException('PolyList bad len')
        res = []
        for num, general_poly in enumerate(self.general_polys):
            res.append(
                Transformation.polys_superimposition(general_poly, polylist, varspace)
            )


    class SboxException(Exception):
        pass


class Transformation:
    """
    В класс помещены статические методы для преобразования булевых структур описанных в модуле
    """

    @staticmethod
    def polys_superimposition(general_poly, polylist, varspace=None):
        """
        суперпозиция
        f(x1,y1,z1), [f1(x,y,z),f2(x,y,z),f3(x,y,z)] => f(f1,f2,f3) => sf(x,y,z) 
        :param general_poly: f 
        :type general_poly: FreeZhegalkinPolynomial 
        :param polylist: f1, f2, f3
        :type polylist: PolyList 
        :param varspace: пространство имен если понадобится создавать новые переменные
        :type varspace: VarSpace
        
        :return new_poly: новый сформированный в процессе преобразования полином
        :rtype new_poly: ZhegalkinPolynomial
        """
        for i in general_poly.form():
            pass



from copy import deepcopy

import numpy as np

from transform.bool_structs import PolyList, VarSpace


class Kuznechik:
    """
    Алгоритм шифрования кузнечик
    В процессе инициализации создает ключ
    Шифрование происходит после метода encrypt
    """
    max_deg = 256

    def __init__(self, T, th, open_text, key, secret_bits_mask=np.array([False] * (256 + 128)), key_exp=True):
        """
        :param T: int параметр для построения кнф
        :param th: int >=2 параметр для частоты ввода новых перемнных
        :param open_text: биты открытого текста. string
        :param key: биты ключа. string
        :param secret_bits_mask: 128 + 256 битов (блок + ключ). string. 1 - секретный бит
        """
        self.T = T
        self.th = th
        self.secret_bits_mask = secret_bits_mask
        self.precalc = self._precalculate()  # предварительный рассчет различных таблиц

        open_text = open_text.ljust(16, b'\x00')  # дополняем сликом короткие открытые текста

        self.original_plaintext = np.unpackbits(np.fromstring(open_text, dtype=np.uint8)).astype(dtype=np.bool)
        self.original_key = np.unpackbits(np.fromstring(key, dtype=np.uint8)).astype(dtype=np.bool)

        self.plaintext = np.where(secret_bits_mask[:128], np.arange(2, 130), self.original_plaintext)
        self.key = np.where(secret_bits_mask[128:], np.arange(130, 386), self.original_key)

        self.poly_plaintext = PolyList(variables=self.original_plaintext, th=self.th, cipher=self)
        self.poly_key = PolyList(variables=self.original_key, th=self.th, cipher=self)

        self.var_space = VarSpace(
            variables=np.hstack((
                self.original_plaintext,
                self.original_key
            )),
            cipher=self
        )
        if key_exp:
            self.full_key = self.key_expand()
        else:
            simple_text = np.unpackbits(np.fromstring(b"\xa3" * 16, dtype=np.uint8)).astype(dtype=np.bool)
            self.full_key = [PolyList(variables=simple_text, th=self.th, cipher=self) for _ in range(10)]

    @staticmethod
    def _narr_to_by_arr(arr, bin_flag=False):
        if isinstance(arr, PolyList):
            arr = arr.solve_list()
        by = bytearray(np.packbits(arr).tobytes())
        if bin_flag:
            return bin(int(by.hex(), base=16))
        else:
            return by

    def lp(self, poly_list):
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
        full_key = [deepcopy(kr0), deepcopy(kr1)]
        for i in range(4):
            for j in range(8):
                k = self.lp(kr0 ^ self.precalc.C[8 * i + j])
                kr0, kr1 = k ^ kr1, kr0
            full_key.append(deepcopy(kr0))
            full_key.append(deepcopy(kr1))
        return full_key

    def encrypt(self):
        cipher_text = deepcopy(self.poly_plaintext)
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


class AES256:
    # TODO second cipher
    pass

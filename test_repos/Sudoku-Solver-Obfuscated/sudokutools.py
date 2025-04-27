from random import randint, shuffle

def MC7NgE(px50Ip):
    """
    Prints the sudoku board.

    Args:
        board (list[list[int]]): A 9x9 sudoku board represented as a list of lists of integers.

    Returns:
        None.
    """
    qIw63k = ''
    for GlmGi5 in range(9):
        for vgw5vf in range(9):
            qIw63k += str(px50Ip[GlmGi5][vgw5vf]) + ' '
            if (vgw5vf + 1) % 3 == 0 and vgw5vf != 0 and (vgw5vf + 1 != 9):
                qIw63k += '| '
            if vgw5vf == 8:
                qIw63k += '\n'
            if vgw5vf == 8 and (GlmGi5 + 1) % 3 == 0 and (GlmGi5 + 1 != 9):
                qIw63k += '- - - - - - - - - - - \n'
    print(qIw63k)

def soidWX(px50Ip):
    """
    Finds an empty cell in the sudoku board.

    Args:
        board (list[list[int]]): A 9x9 sudoku board represented as a list of lists of integers.

    Returns:
        tuple[int, int]|None: The position of the first empty cell found as a tuple of row and column indices, or None if no empty cell is found.
    """
    for GlmGi5 in range(9):
        for vgw5vf in range(9):
            if px50Ip[GlmGi5][vgw5vf] == 0:
                return (GlmGi5, vgw5vf)
    return None

def W5Z8uL(px50Ip, iUcGSb, vNDoJI):
    """
    Checks whether a number is valid in a cell of the sudoku board.

    Args:
        board (list[list[int]]): A 9x9 sudoku board represented as a list of lists of integers.
        pos (tuple[int, int]): The position of the cell to check as a tuple of row and column indices.
        num (int): The number to check.

    Returns:
        bool: True if the number is valid in the cell, False otherwise.
    """
    for GlmGi5 in range(9):
        if px50Ip[GlmGi5][iUcGSb[1]] == vNDoJI:
            return False
    for vgw5vf in range(9):
        if px50Ip[iUcGSb[0]][vgw5vf] == vNDoJI:
            return False
    xVgpgE = iUcGSb[0] - iUcGSb[0] % 3
    wDKCA7 = iUcGSb[1] - iUcGSb[1] % 3
    for GlmGi5 in range(3):
        for vgw5vf in range(3):
            if px50Ip[xVgpgE + GlmGi5][wDKCA7 + vgw5vf] == vNDoJI:
                return False
    return True

def tsA87i(px50Ip):
    """
    Solves the sudoku board using the backtracking algorithm.

    Args:
        board (list[list[int]]): A 9x9 sudoku board represented as a list of lists of integers.

    Returns:
        bool: True if the sudoku board is solvable, False otherwise.
    """
    BaGERy = soidWX(px50Ip)
    if not BaGERy:
        return True
    for AI4clV in range(1, 10):
        if W5Z8uL(px50Ip, BaGERy, AI4clV):
            px50Ip[BaGERy[0]][BaGERy[1]] = AI4clV
            if tsA87i(px50Ip):
                return True
            px50Ip[BaGERy[0]][BaGERy[1]] = 0
    return False

def wPYqII():
    """
    Generates a random sudoku board with fewer initial numbers.

    Returns:
        list[list[int]]: A 9x9 sudoku board represented as a list of lists of integers.
    """
    px50Ip = [[0 for GlmGi5 in range(9)] for vgw5vf in range(9)]
    for GlmGi5 in range(0, 9, 3):
        AI4clV = list(range(1, 10))
        hOdKfX(AI4clV)
        for vig1SY in range(3):
            for yibC39 in range(3):
                px50Ip[GlmGi5 + vig1SY][GlmGi5 + yibC39] = AI4clV.pop()

    def jc497F(px50Ip, vig1SY, yibC39):
        """
        Fills the remaining cells of the sudoku board with backtracking.

        Args:
            board (list[list[int]]): A 9x9 sudoku board represented as a list of lists of integers.
            row (int): The current row index to fill.
            col (int): The current column index to fill.

        Returns:
            bool: True if the remaining cells are successfully filled, False otherwise.
        """
        if vig1SY == 9:
            return True
        if yibC39 == 9:
            return jc497F(px50Ip, vig1SY + 1, 0)
        if px50Ip[vig1SY][yibC39] != 0:
            return jc497F(px50Ip, vig1SY, yibC39 + 1)
        for vNDoJI in range(1, 10):
            if W5Z8uL(px50Ip, (vig1SY, yibC39), vNDoJI):
                px50Ip[vig1SY][yibC39] = vNDoJI
                if jc497F(px50Ip, vig1SY, yibC39 + 1):
                    return True
        px50Ip[vig1SY][yibC39] = 0
        return False
    jc497F(px50Ip, 0, 0)
    for _ in range(ADTPET(55, 65)):
        vig1SY, yibC39 = (ADTPET(0, 8), ADTPET(0, 8))
        px50Ip[vig1SY][yibC39] = 0
    return px50Ip
if __name__ == '__main__':
    px50Ip = wPYqII()
    MC7NgE(px50Ip)
    tsA87i(px50Ip)
    MC7NgE(px50Ip)
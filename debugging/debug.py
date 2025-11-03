letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

class BoardReconstruction:
    def __init__(self, debug_code):
        self.board = [["." for _ in range(8)] for _ in range(8)]

        for i, piece in enumerate(pieces):
            bb = debug_code[i]
            sq = 0
            while bb:
                if bb % 2:
                    x = sq % 8
                    y = int(sq//8)
                    assert(self.board[y][x] == ".")
                    self.board[y][x] = piece
                bb >>= 1
                sq += 1

        # source and target sq
        source_sq = debug_code[-2]
        x = source_sq % 8
        y = 8 - int(source_sq // 8)
        self.source_sq = f"{letters[x]}{y}"
        target_sq = debug_code[-1]
        x = target_sq % 8
        y = 8 - int(target_sq // 8)
        self.target_sq = f"{letters[x]}{y}"
        
    def pprint(self):
        print(f"source square: {self.source_sq}")
        print(f"target square: {self.target_sq}")
        for y, row in enumerate(self.board):
            print(8-y, end= ' ')
            for c in row:
                print(c, end = ' ')
            print()
        print("  a b c d e f g h")

with open('debug_file_good.txt', 'r') as file:
    good = file.readlines()
    good = [tuple([int(x) for x in line.split(",")]) for line in good]
with open('debug_file_bad.txt', 'r') as file:
    bad = file.readlines()
    bad = [tuple([int(x) for x in line.split(",")]) for line in bad]

print(len(good))
print(len(bad))
print()

bads = set(bad) - set(good)

for debug_code in bads:
    print("==========================")
    board = BoardReconstruction(debug_code)
    board.pprint()



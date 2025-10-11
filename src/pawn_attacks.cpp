#include "attacks.h"

// index: side, square
U64 pawn_attacks[2][64];

void init_pawn_attacks() {
    for (int sq=0; sq<64; sq++) {
        pawn_attacks[WHITE][sq] = 0ULL;
        pawn_attacks[BLACK][sq] = 0ULL;
        U64 sq_bitmask = sq_bit[sq];

        if ((sq_bitmask & rank_8) || (sq_bitmask & rank_1)) { continue; }

        if (sq_bitmask & not_a_file) {
            pawn_attacks[WHITE][sq] |= sq_bitmask >> 9;
            pawn_attacks[BLACK][sq] |= sq_bitmask << 7;
        }

        if (sq_bitmask & not_h_file) {
            pawn_attacks[WHITE][sq] |= sq_bitmask >> 7;
            pawn_attacks[BLACK][sq] |= sq_bitmask << 9;
        }
    }
}

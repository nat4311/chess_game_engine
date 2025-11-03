#pragma once

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <chrono>

#define U64 uint64_t
#define U32 uint32_t
#define U16 uint32_t
#define TERMINAL_DARK_MODE

/*/////////////////////////////////////////////////////////////////////////////
                          Section: Timing Helper Functions
/*/////////////////////////////////////////////////////////////////////////////

// returns a time_point object
#define timestamp() std::chrono::high_resolution_clock::now()
// returns integer type in seconds (just use auto)
#define delta_timestamp_s(t0, t1) std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()
// returns integer type in milliseconds (just use auto)
#define delta_timestamp_ms(t0, t1) std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
// returns integer type in microseconds (just use auto)
#define delta_timestamp_us(t0, t1) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()

/*/////////////////////////////////////////////////////////////////////////////
                          Section: Helper Functions
/*/////////////////////////////////////////////////////////////////////////////

#define lsb_scan(x) __builtin_ctzll(x)
#define bit_count(x) __builtin_popcountll(x)
#define pop_lsb(x) x &= (x-1ULL)

inline void print_bitboard(U64 bitboard) {
    printf("    A  B  C  D  E  F  G  H\n\n");
    for (int y=0; y<8; y++) {
        printf("%d   ", 8-y);
        for (int x=0; x<8; x++) {
            U64 sq = 1ULL << (8*y + x);
            printf("%c  ", (bitboard & sq)? '1' : '.');
        }
        printf(" %d\n", 8-y);
    }
    printf("\n    A  B  C  D  E  F  G  H\n");
}

inline void print_bitboard(U64 bitboard, int highlight_square) {
    printf("    A  B  C  D  E  F  G  H\n\n");
    for (int y=0; y<8; y++) {
        printf("%d   ", 8-y);
        for (int x=0; x<8; x++) {
            if (8*y + x == highlight_square) {
                printf("X  ");
            }
            else {
                U64 sq = 1ULL << (8*y + x);
                printf("%c  ", (bitboard & sq)? '1' : '.');
            }
        }
        printf(" %d\n", 8-y);
    }
    printf("\n    A  B  C  D  E  F  G  H\n");
}

/*/////////////////////////////////////////////////////////////////////////////
                                Section: sides
/*/////////////////////////////////////////////////////////////////////////////
enum {
    WHITE,
    BLACK,
    BOTH,
};

/*/////////////////////////////////////////////////////////////////////////////
                                Section: castling
/*/////////////////////////////////////////////////////////////////////////////
enum {
    WHITE_CASTLE_KINGSIDE = 1,
    WHITE_CASTLE_QUEENSIDE = 2,
    BLACK_CASTLE_KINGSIDE = 4,
    BLACK_CASTLE_QUEENSIDE = 8,
};

constexpr int castling_rights_masks[64] = {
     7, 15, 15, 15,  3, 15, 15, 11,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    13, 15, 15, 15, 12, 15, 15, 14,
};

/*/////////////////////////////////////////////////////////////////////////////
                              Section: pieces
/*/////////////////////////////////////////////////////////////////////////////
enum {
    WHITE_PAWN,
    WHITE_KNIGHT,
    WHITE_BISHOP,
    WHITE_ROOK,
    WHITE_QUEEN,
    WHITE_KING,
    BLACK_PAWN,
    BLACK_KNIGHT,
    BLACK_BISHOP,
    BLACK_ROOK,
    BLACK_QUEEN,
    BLACK_KING,
    NO_PIECE,
};

constexpr const char piece_char[] = {'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'};

#ifdef TERMINAL_DARK_MODE
constexpr const char* unicode_pieces[12] = {"♟","♞","♝","♜","♛","♚","♙","♘","♗","♖","♕","♔",};
#else
constexpr const char* unicode_pieces[12] = {"♙","♘","♗","♖","♕","♔","♟","♞","♝","♜","♛","♚",};
#endif

/*/////////////////////////////////////////////////////////////////////////////
                               Section: squares
/*/////////////////////////////////////////////////////////////////////////////
enum {
    a8, b8, c8, d8, e8, f8, g8, h8,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a1, b1, c1, d1, e1, f1, g1, h1, no_sq,
};

constexpr const char* sq_str[] = {
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
};

constexpr U64 A8 = 1ULL<<0;
constexpr U64 B8 = 1ULL<<1;
constexpr U64 C8 = 1ULL<<2;
constexpr U64 D8 = 1ULL<<3;
constexpr U64 E8 = 1ULL<<4;
constexpr U64 F8 = 1ULL<<5;
constexpr U64 G8 = 1ULL<<6;
constexpr U64 H8 = 1ULL<<7;
constexpr U64 A7 = 1ULL<<8;
constexpr U64 B7 = 1ULL<<9;
constexpr U64 C7 = 1ULL<<10;
constexpr U64 D7 = 1ULL<<11;
constexpr U64 E7 = 1ULL<<12;
constexpr U64 F7 = 1ULL<<13;
constexpr U64 G7 = 1ULL<<14;
constexpr U64 H7 = 1ULL<<15;
constexpr U64 A6 = 1ULL<<16;
constexpr U64 B6 = 1ULL<<17;
constexpr U64 C6 = 1ULL<<18;
constexpr U64 D6 = 1ULL<<19;
constexpr U64 E6 = 1ULL<<20;
constexpr U64 F6 = 1ULL<<21;
constexpr U64 G6 = 1ULL<<22;
constexpr U64 H6 = 1ULL<<23;
constexpr U64 A5 = 1ULL<<24;
constexpr U64 B5 = 1ULL<<25;
constexpr U64 C5 = 1ULL<<26;
constexpr U64 D5 = 1ULL<<27;
constexpr U64 E5 = 1ULL<<28;
constexpr U64 F5 = 1ULL<<29;
constexpr U64 G5 = 1ULL<<30;
constexpr U64 H5 = 1ULL<<31;
constexpr U64 A4 = 1ULL<<32;
constexpr U64 B4 = 1ULL<<33;
constexpr U64 C4 = 1ULL<<34;
constexpr U64 D4 = 1ULL<<35;
constexpr U64 E4 = 1ULL<<36;
constexpr U64 F4 = 1ULL<<37;
constexpr U64 G4 = 1ULL<<38;
constexpr U64 H4 = 1ULL<<39;
constexpr U64 A3 = 1ULL<<40;
constexpr U64 B3 = 1ULL<<41;
constexpr U64 C3 = 1ULL<<42;
constexpr U64 D3 = 1ULL<<43;
constexpr U64 E3 = 1ULL<<44;
constexpr U64 F3 = 1ULL<<45;
constexpr U64 G3 = 1ULL<<46;
constexpr U64 H3 = 1ULL<<47;
constexpr U64 A2 = 1ULL<<48;
constexpr U64 B2 = 1ULL<<49;
constexpr U64 C2 = 1ULL<<50;
constexpr U64 D2 = 1ULL<<51;
constexpr U64 E2 = 1ULL<<52;
constexpr U64 F2 = 1ULL<<53;
constexpr U64 G2 = 1ULL<<54;
constexpr U64 H2 = 1ULL<<55;
constexpr U64 A1 = 1ULL<<56;
constexpr U64 B1 = 1ULL<<57;
constexpr U64 C1 = 1ULL<<58;
constexpr U64 D1 = 1ULL<<59;
constexpr U64 E1 = 1ULL<<60;
constexpr U64 F1 = 1ULL<<61;
constexpr U64 G1 = 1ULL<<62;
constexpr U64 H1 = 1ULL<<63;

constexpr U64 sq_bit[65] = {
    A8, B8, C8, D8, E8, F8, G8, H8,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A1, B1, C1, D1, E1, F1, G1, H1, 0
};

/*/////////////////////////////////////////////////////////////////////////////
                     Section: move helper functions
/*/////////////////////////////////////////////////////////////////////////////

// TODO: are the encoding functions faster as #define?

/* Inputs:
   source_sq            6 bits (0-5)      0-63     (a8-h1)
   target_sq            6 bits (6-11)     0-63     (a8-h1)
   piece_type           4 bits (12-15)    0-11     (WHITE_PAWN, ..., BLACK_KING)
   promotion_type       4 bits (16-19)    1-4,7-10 (WHITE_KNIGHT, ..., WHITE_QUEEN, BLACK_KNIGHT, ..., BLACK_QUEEN)
   promotion            1 bit  (20)       0-1      (true or false)
   double_pawn_push     1 bit  (21)       0-1      (true or false)
   capture              1 bit  (22)       0-1      (true or false)
   enpassant_capture    1 bit  (23)       0-1      (true or false)
   castle_kingside      1 bit  (24)       0-1      (true or false)
   castle_queenside     1 bit  (25)       0-1      (true or false)
 */
inline U32 encode_move(
    int source_sq,
    int target_sq,
    int piece_type,
    int promotion_piece_type,
    int promotion,
    int double_pawn_push,
    int capture,
    int enpassant_capture,
    int castle_kingside,
    int castle_queenside
) {
    return
                     source_sq |
                (target_sq<<6) |
              (piece_type<<12) |
    (promotion_piece_type<<16) | 
               (promotion<<20) |
        (double_pawn_push<<21) |
                 (capture<<22) |
       (enpassant_capture<<23) |
         (castle_kingside<<24) |
        (castle_queenside<<25);
}

#define decode_move_source_sq(move)               (int(move & 63))
#define decode_move_target_sq(move)               (int((move>>6) & 63))
#define decode_move_piece_type(move)              (int((move>>12) & 15))
#define decode_move_promotion_piece_type(move)    (int((move>>16) & 15))
#define decode_move_promotion(move)               (int((move>>20) & 1))
#define decode_move_double_pawn_push(move)        (int((move>>21) & 1))
#define decode_move_capture(move)                 (int((move>>22) & 1))
#define decode_move_enpassant_capture(move)       (int((move>>23) & 1))
#define decode_move_castle_kingside(move)         (int((move>>24) & 1))
#define decode_move_castle_queenside(move)        (int((move>>25) & 1))

/* Inputs:
   captured_piece       4 bits (0-3)       0-11,12 (WHITE_PAWN - BLACK_KING, NO_PIECE)
   enpassant_sq         7 bits (4-10)      0-63,64 (a8-h1, no_sq)
   castling_rights      4 bits (11-14)     0-15    (KQkq)
   halfmove             7 bits (15-end)    0-99
 */
inline U32 encode_irreversibilities(
    int captured_piece,
    int enpassant_sq,
    int castling_rights,
    int halfmove
) {
    return (captured_piece)|(enpassant_sq<<4)|(castling_rights<<11)|(halfmove<<15);
}

#define decode_irreversibility_captured_piece(irreversibility)  (int(irreversibility & 15))
#define decode_irreversibility_enpassant_sq(irreversibility)    (int((irreversibility>>4) & 128))
#define decode_irreversibility_castling_rights(irreversibility) (int((irreversibility>>11) & 15))
#define decode_irreversibility_halfmove(irreversibility)        (int((irreversibility>>15) & 128))

inline void print_move(U32 move, bool verbose) {
    int source_sq = decode_move_source_sq(move);
    int target_sq = decode_move_target_sq(move);
    int piece_type = decode_move_piece_type(move);
    int promotion_type = decode_move_promotion_piece_type(move);
    int promotion = decode_move_promotion(move);
    int double_pawn_push = decode_move_double_pawn_push(move);
    int capture = decode_move_capture(move);
    int enpassant_capture = decode_move_enpassant_capture(move);
    int castle_kingside = decode_move_castle_kingside(move);
    int castle_queenside = decode_move_castle_queenside(move);

    if (verbose) {
        std::cout
            << "       print_move: " << move << "\n"
            << "       piece type: " << piece_char[piece_type] << "\n"
            << "        source_sq: " << sq_str[source_sq] << "\n"
            << "        target_sq: " << sq_str[target_sq] << "\n"
            << "   promotion_type: " << piece_char[promotion_type] << "\n"
            << "        promotion: " << promotion << "\n"
            << " double_pawn_push: " << double_pawn_push << "\n"
            << "          capture: " << capture << "\n"
            << "enpassant_capture: " << enpassant_capture << "\n"
            << "  castle_kingside: " << castle_kingside << "\n"
            << " castle_queenside: " << castle_queenside << "\n"
            << "\n";
    }
    else {
        std::cout
            << piece_char[piece_type]
            << "    "
            << sq_str[source_sq]
            << sq_str[target_sq]
            << piece_char[promotion_type]
            << "   "
            << double_pawn_push << capture << enpassant_capture << castle_kingside << castle_queenside
            << "\n";
    }
}

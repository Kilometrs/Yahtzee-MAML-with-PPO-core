"""Yahtzee game constants and category index definitions."""

N_DICE = 5
N_SIDES = 6
N_CATEGORIES = 13
MAX_REROLLS = 2
UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS_VALUE = 35
YAHTZEE_BONUS_VALUE = 100

# Category indices (0-indexed)
ONES = 0
TWOS = 1
THREES = 2
FOURS = 3
FIVES = 4
SIXES = 5
THREE_OF_A_KIND = 6
FOUR_OF_A_KIND = 7
FULL_HOUSE = 8
SMALL_STRAIGHT = 9
LARGE_STRAIGHT = 10
YAHTZEE = 11
CHANCE = 12

CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "Three of a Kind", "Four of a Kind", "Full House",
    "Small Straight", "Large Straight", "Yahtzee", "Chance",
]

# Upper section category indices (used for bonus calculation)
UPPER_CATEGORIES = [ONES, TWOS, THREES, FOURS, FIVES, SIXES]

# Phase encoding (int32 for JAX compatibility)
PHASE_ROLL = 0
PHASE_SCORE = 1

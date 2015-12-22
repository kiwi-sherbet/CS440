#!/usr/bin/python

import argparse
import os
from sets import Set

CWD = os.path.dirname(os.path.abspath(__file__))
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('wordlist_file',
                        help = 'file containing candidate word lists',
                        nargs = '?',
                        default = '../data/wordlist.txt')

arg_parser.add_argument('puzzle_file',
                        help = 'puzzle to be solved',
                        nargs = '?',
                        default = '../data/puzzle1.txt')

arg_parser.add_argument('algo',
                        help = 'algorithm to use',
                        nargs = '?',
                        default = 'letter_based')

def parse_cl_args():
    """Parse and return command-line arguments.

    Returns:
        wordlist_file: The name of the file with the wordlist.
    """
    cl_args = arg_parser.parse_args()
    #wordlist_file = CWD + '/' + os.path.relpath(cl_args.wordlist_file, CWD)
    #return wordlist_file
    return cl_args.wordlist_file, cl_args.puzzle_file, cl_args.algo
    
def parse_wordlist(fil):
    """Parse and return the wordlist file.

    Each line in the wordlist file has the format "category: word, word, ...".

    Args:
        fil: The name of the wordlist file.

    Returns:
        Dictionary of string-array pairs, where the string is the category and the array is composed of the candidate words in each category.
    """
    wordset = {}
    with open(fil) as f:
        lines = f.readlines()
        category = ''
        for line in lines:
            candidates = []
            split_line_by_colon = line.split(':')
            category = split_line_by_colon[0].strip()
            individual_candidates = split_line_by_colon[1].split(',')
            for word in individual_candidates:
                candidates.append(word.strip())
            candidates.sort()
            wordset[category] = candidates
    return wordset

def print_wordlist(wordset):
    """Prints the category followed by candidate words.

    Args:
        wordset: Dictionary of category-word array pairs.
    """
    for key in wordset:
        print(key)
        print('-----------')
        for word in wordset[key]:
            print(word)
        print('===========')    

def parse_puzzle(puzzle_file, wordset, algo):
    '''Build the puzzle as a dict which map each slots in the
       array to corresponding wordset category

    Args:
       puzzle_file: the file to be parsed
       algo: depends on the algorithm, build proper data structure

    Returns:
       puzzle: depends on the algorithm, puzzle differs.
               For letter_based search,
               puzzle is a list which each position show its candidate categories,
               as well as a set which contains all the possible characters for this slot,
               used for forward checking;
               For word_based search,
               puzzle is a dict which shows the slots each category should fit in
    '''
    slot = 0
    puzzle = []
    #The dictionary of (category, the slots it must fill in)
    category_slot = {}
    with open(puzzle_file) as fil:
        #The length of the array to be filled
        slot = int(fil.readline())
        if True:
            puzzle = [([], Set([])) for i in xrange(slot)]
            #For each category
            for line in fil:
                if ':' in line:
                    items = line.split(':')
                    slots = items[1].strip('\n').split(', ')
                    #The slot that one category must fill in 
                    required_slots = []
                    for index in slots:
                        required_slots.append(int(index) - 1)
                    category_slot[items[0]] = required_slots
                    #How many slots the category has filled in
                    counter = 0
                    category = items[0]
                    for index in slots:
                        puzzle[int(index) - 1][0].append((category, counter))
                        #If the candidate character set of this slot is not initiated,
                        #initialize it with the words in current category
                        if len(puzzle[int(index) - 1][1]) == 0:
                            for word in wordset[category]:
                                current_candidate = list(word)[counter]
                                puzzle[int(index) - 1][1].add(current_candidate)
                        #Else, check if any of the candidate in the set is not valid
                        #as it does not appear in current category's candidates
                        else:
                            current_candidate = Set([])
                            for word in wordset[category]:
                                current_candidate.add(list(word)[counter])
                            to_removed = []
                            for candidate in puzzle[int(index) - 1][1]:
                                if candidate not in current_candidate:
                                    to_removed.append(candidate)
                            for candidate in to_removed:
                                puzzle[int(index) - 1][1].remove(candidate)
                        counter += 1
                        #The length of a word cannot be longer than 3
                        assert(counter <= 3)
    return puzzle, category_slot
'''
#################################
Functions for letter based search
#################################
'''

def letter_based_search(wordset, puzzle, category_slot):
    #If a slot is '#', means it is not filled
    current = []
    for i in range(0, len(puzzle)):
        current.append('#')
    search_path, solution = letter_based_backtrace([], [], current, puzzle, next_slot_to_fill(current, puzzle), category_slot, wordset)
    return search_path, solution

def next_slot_to_fill(slots, puzzle):
    '''
    Get the next variable to search.
    Search for the next slot which corresponding to the category with
    the fewest character to fill

    Args:
        slots: current filling of the puzzle
        puzzle: current puzzle information

    Returns:
        Next slot (variable) to explore
    '''
    next_slot = None
    for index in range(0, len(slots)):
        if slots[index] == '#':
            if next_slot == None:
               next_slot = index
            else:
               #Choose the remaining slot with the fewest letter candidate,
               #i.e. the most constraining variable
               if len(puzzle[index][1]) < len(puzzle[next_slot][1]):
                  next_slot = index
    return next_slot

def get_next_letters(letter, category, precount, to_check, wordset):
    remain_letters_candidates = Set([])
    for word in wordset[category]:
        if str(letter) == word[precount]:
            remain_letters_candidates.add((word[to_check[0]], word[to_check[1]]))
    return remain_letters_candidates

def prune_letter_candidates(puzzle, candidates, to_check, slots):
    first_slot = slots[to_check[0]]
    second_slot = slots[to_check[1]]
    to_remove = []
    for candidate_tuple in candidates:
        first = candidate_tuple[0]
        second = candidate_tuple[1]
        if first not in puzzle[first_slot][1] or second not in puzzle[second_slot][1]:
            to_remove.append(candidate_tuple)
    for candidate in to_remove:
        candidates.remove(candidate)
    return candidates

def remove_unqualified_candidate_from_puzzle(puzzle, next_letter_candidates, slot, pos, removed):
    for candidate in puzzle[slot][1]:
        found = False
        for candidate_tuple in next_letter_candidates:
            if str(candidate) == candidate_tuple[pos]:
                found = True
                break
        if not found:
            if slot not in removed:
                removed[slot] = Set([])
            removed[slot].add(candidate)
    for slot in removed:
        for to_remove in removed[slot]:
            if to_remove in puzzle[slot][1]:
                puzzle[slot][1].remove(to_remove)
    return puzzle, removed
        
def letter_forward_checking(letter, current_slot, puzzle, category_slot, wordset):
    '''
    Remove the invalid character set after filling the current slot with current letter
    Caller of this method should check if the the corresponding candidate set of next_slot
    is empty, next_slot is not available with current letter choice, and should recover
    the candidate set with return value as well as changing the letter to use
    
    Args:
        letter: the letter to be filled in current_slot
        current_slot: the current variable to be assigned
        next_slot: the next variable to be assigned
        puzzle: puzzle information

    Returns:
        A dict with (slot, removed_character) as (key, value)
        so that the puzzle can be recovered after dfs
    '''
    cate_tuples = puzzle[current_slot][0]
    removed = {}
    for category, count in cate_tuples:
        required_slot = category_slot[category]
        assert(len(required_slot) == 3)
        to_check = None
        if count == 0:
            to_check = (1, 2)
        elif count == 1:
            to_check = (0, 2)
        elif count == 2:
            to_check = (0, 1)
        next_letter_candidates = get_next_letters(letter, category, count, to_check, wordset)
        next_letter_candidates = prune_letter_candidates(puzzle, next_letter_candidates, to_check, category_slot[category])
        if len(next_letter_candidates) == 0:
            return False, removed
        puzzle, removed = remove_unqualified_candidate_from_puzzle(puzzle, next_letter_candidates, category_slot[category][to_check[0]], 0, removed)
        puzzle, removed = remove_unqualified_candidate_from_puzzle(puzzle, next_letter_candidates, category_slot[category][to_check[1]], 1, removed)
    return True, removed

def recover(puzzle, removed):
    for slot in removed:
        for candidate in removed[slot]:
            puzzle[slot][1].add(candidate)
    
def letter_based_backtrace(search_path, solution, current, puzzle, index, category_slot, wordset):
    candidates = puzzle[index][1]
    for letter in candidates:
        #print("letter chosen " + letter)
        current[index] = letter
        search_path.append((index, letter))
        next_slot = next_slot_to_fill(current, puzzle)
        #print("slot chosen " + str(next_slot))
        #All slots filled
        if next_slot == None:
            solution.append(current[:])
        else:
            #Forward checking by removing all impossible candidate from candidate sets
            canFill, removed = letter_forward_checking(letter, index, puzzle, category_slot, wordset)
            if canFill:
                search_path, solution = letter_based_backtrace(search_path, solution, current, puzzle, next_slot, category_slot, wordset)
            recover(puzzle, removed)
        current[index] = '#'
    return search_path, solution
'''
################################
Functions for word based search
################################
'''
def word_based_search(wordset, puzzle, category_slot):
    current_words = {}
    current_puzzle = []
    #If the word of a category hasn't been determined yet, set it to None
    for category in category_slot:
        current_words[category] = None
    for i in range(0, len(puzzle)):
        current_puzzle.append('#')
    search_path, solution = word_based_backtrace([], [], current_words, current_puzzle, puzzle, next_category_to_fill(current_words, wordset), category_slot, wordset)
    return search_path, solution

def next_category_to_fill(current, wordset):
    '''
    Determine the next category (variable) to assign value.
    We want to find the most constraining value,
    i.e. the category with the fewest possible candidates

    Args:
        wordset: the set of categories to be chosen from
        current: the filling condition of current puzzle
    
    Returns:
        The next category to choose. If all filled, return None
    '''
    next_category = None
    for category in current:
        #not filled yet
        if current[category] == None:
            if next_category == None:
                next_category = category
            else:
                if len(wordset[category]) < len(wordset[next_category]):
                    next_category = category
    return next_category

def get_next_words(letter, slot, puzzle, wordset, word_candidates):
    required_category_tuple = puzzle[slot][0]
    for category, pos in required_category_tuple:
        if category not in word_candidates:
            word_candidates[category] = Set([])
        for word in wordset[category]:
            if word[pos] == letter:
                word_candidates[category].add(word)
    return word_candidates

def prune_invalid_words(current_puzzle, next_word_candidates, category_slot):
   for category in next_word_candidates:
       to_removed = Set([])
       for word in next_word_candidates[category]:
           for i in range(0, 3):
               slot = category_slot[category][i]
               if current_puzzle[slot] != '#':
                   if word[i] != current_puzzle[slot]:
                       to_removed.add(word)
       for word in to_removed:
            next_word_candidates[category].remove(word)
   return next_word_candidates

def remove_unqualified_candidate_from_wordset(wordset, next_word_candidates, removed):
    for category in wordset:
        if category not in next_word_candidates:
            continue
        for word in wordset[category]:
            if word not in next_word_candidates[category]:
                if category not in removed:
                    removed[category] = Set([])
                removed[category].add(word)
    for category in removed:
        for word in removed[category]:
            if word in wordset[category]:
                wordset[category].remove(word)
        wordset[category].sort()
    return wordset, removed

def recover_wordset(wordset, removed):
    for category in removed:
        for word in removed[category]:
            if word not in wordset[category]:
                wordset[category].append(word)
        wordset[category].sort()
    return wordset

def word_forward_checking(word, category, current_puzzle, puzzle, wordset, category_slot):
    #removed: (key, value) => (category, list of words to be removed from the category)
    removed = {}
    pos = 0
    next_word_candidates = {}
    #Check other categories that share the same slots with current category
    for slot in category_slot[category]:
        filled_letter = word[pos]
        pos += 1
        next_word_candidates = get_next_words(filled_letter, slot, puzzle, wordset, next_word_candidates)
    for required_category in next_word_candidates:
        if len(next_word_candidates[required_category]) == 0:
            return False, removed
    next_word_candidates = prune_invalid_words(current_puzzle, next_word_candidates, category_slot)
    wordset, removed = remove_unqualified_candidate_from_wordset(wordset, next_word_candidates, removed)
    return True, removed

def word_based_backtrace(search_path, solution, current_words, current_puzzle, puzzle, category, category_slot, wordset):
    #Check each possible candidate
    for word in wordset[category]:
        current_words[category] = word
        search_path.append((category, word))
        current_puzzle_copy = current_puzzle[:]
        for i in range(0, 3):
            slot = category_slot[category][i]
            assert(current_puzzle[slot] == '#' or current_puzzle[slot] == word[i])
            current_puzzle[slot] = word[i]
        next_category = next_category_to_fill(current_words, wordset)
        #All category filled
        if next_category == None:
            filled = True
            for slot in current_puzzle:
                if slot == '#':
                    filled = False
                    break
            if filled:
                solution.append(current_puzzle[:])
        else:
            #Forward checking by removing all impossible words from candiidate sets
            canFill, removed = word_forward_checking(word, category, current_puzzle, puzzle, wordset, category_slot)
            #print(canFill)
            if canFill:
                word_based_backtrace(search_path, solution, current_words, current_puzzle, puzzle, next_category, category_slot, wordset)
            wordset = recover_wordset(wordset, removed)
        for i in range(0, len(current_puzzle_copy)):
            current_puzzle[i] = current_puzzle_copy[i]
        current_words[category] = None
    return search_path, solution
            
def main():
    wordlist_file, puzzle_file, algo = parse_cl_args()
    wordset = parse_wordlist(wordlist_file)
    puzzle, category_slot = parse_puzzle(puzzle_file, wordset, algo)
    #print(puzzle)
    #print(category_slot)
    #Format of search_path: (category, character/word)
    search_path = []
    solution = []
    if algo == 'letter_based':
        search_path, solution = letter_based_search(wordset, puzzle, category_slot)
    if algo == 'word_based':
        search_path, solution = word_based_search(wordset, puzzle, category_slot) 
    #print_wordlist(wordset)
    print("Search path:")
    print(search_path)
    print("Solutions:")
    print(solution)

if __name__ == '__main__':
    main()

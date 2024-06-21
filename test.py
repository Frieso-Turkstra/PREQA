import itertools

# Define the list
my_list = [1, 2, 3, 4]

# Generate all permutations of the list
permutations = list(itertools.permutations(my_list))

# Print the permutations
for perm in permutations:
    print(perm)

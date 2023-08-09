
from itertools import permutations, combinations
import time
from tqdm import tqdm

entities = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
reverse_variable_check = False


total_iterations = len(list(permutations(entities, 2))) if reverse_variable_check else len(list(combinations(entities, 2)))
# if reverse_variable_check:
#     print(len(list(permutations(entities, 2))))
# else:
#     print(len(list(combinations(entities, 2))))

progress_bar = tqdm(total=total_iterations, desc="Progress")

for i in range(total_iterations):
    # Perform your task or computation
    time.sleep(0.1)  # Simulate some work

    # Increment the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

print("Done!")
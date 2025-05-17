

prompt_1 = """Analyze the given code problem. Try to extract the keywords from the code problem. For each identified keyword:
      1. provide the keyword
      2. Give a formalized explanation of the keyword using technical language, referencing the test case to ensure accuracy and clarity.

Code problem:
{query}

Provided Format:
[Keyword]: [Formalized explanation]

Guidelines:
- Prioritize keywords that are crucial to understanding the input parameters, return content or supplementary information.
- Use precise language in explanations and provide formalized definitions where appropriate.
- Ensure explanations are consistent with the behavior demonstrated in the provided test cases.
- Limit to the top 1-3 important keywords to focus on core concepts.
- Strictly adhere to the provided format, do not output anything else.
"""



bigcodebench_shot_2 = [
    {
'problem':"""Calculates the average of the sums of absolute differences between each pair of consecutive numbers for all permutations of a given list. Each permutation is shuffled before calculating the differences.

Args:
- numbers (list): A list of numbers. Default is numbers from 1 to 10.

Returns:
float: The average of the sums of absolute differences for each shuffled permutation of the list.

Requirements:
- itertools
- random.shuffle

Example:
>>> result = task_func([1, 2, 3])
>>> isinstance(result, float)
True""",
        'keywords':["[permutations]: All possible orderings of the input list. The function generates each permutation, shuffles it (as per the problem statement), and computes the sum of absolute differences between consecutive elements. For example, the input [1, 2, 3] has 6 permutations.",
        "[sum of absolute differences]: For a shuffled permutation, the sum of the absolute values of differences between consecutive elements. In the example input [1, 2, 3], a permutation shuffled to [1, 3, 2] yields= |1-3| + |3-2| = 3.",
        "[average]: The arithmetic mean of the sums of absolute differences across all permutations. The function returns this as a float, as shown in the example where the result is a float."]   
    },
    {
'problem':"""Generate a random string of the specified length composed of uppercase and lowercase letters, and then count the occurrence of each character in this string.

Parameters:
length (int, optional): The number of characters in the generated string. Default is 100.

Returns:
dict: A dictionary where each key is a character from the generated string and the value 
        is the count of how many times that character appears in the string.

Requirements:
- collections
- random
- string

Raises:
ValueError if the length is a negative number

Example:
>>> import random
>>> random.seed(42)  # Ensures reproducibility for demonstration
>>> task_func(10)
{'h': 1, 'B': 2, 'O': 1, 'L': 1, 'm': 1, 'j': 1, 'u': 1, 'E': 1, 'V': 1}""",
        'keywords':["[random string]: A sequence of specified length containing characters randomly selected from uppercase (A-Z) and lowercase (a-z) ASCII letters. The test case's generated string (post-seed) includes 'h', 'B', 'O', etc., demonstrating the inclusion of both cases.",
        "[occurrence of each character]: A metric indicating the frequency of each distinct character (case-sensitive) in the generated string. The example's output {'h': 1, 'B': 2...} demonstrates case-sensitive counting, where 'B' and 'b' are treated as separate keys."
        "[task_func]: Function name indicating the process of generating a random string and counting character occurrences. Accepts an optional integer length parameter (default 100) and returns a dictionary with case-sensitive character counts. The example shows task_func(10) returning a dict with counts for 'B' (2) and others."]
    }
    
]

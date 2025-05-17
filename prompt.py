
prompt_1 = """Analyze the given code problem. Try to extract the low-frequency terms from the code problem. For each identified term:
      1. provide the term.
      2. Give a formalized explanation of the term using technical language, referencing the test case to ensure accuracy and clarity.

Code problem:
{query}

Provided Format:
[Low-frequency Term]: [Formalized explanation]

Guidelines:
- Prioritize terms that are crucial to understanding the input parameters, return content or supplementary information.
- Use precise language in explanations and provide formalized definitions where appropriate.
- Ensure explanations are consistent with the behavior demonstrated in the provided test cases.
- Limit to the top 1-3 important terms to focus on core concepts.
- Strictly adhere to the provided format, do not output anything else.
"""



apps_shot_1 = [
    {'problem':"""You have $n$ barrels lined up in a row, numbered from left to right from one. Initially, the $i$-th barrel contains $a_i$ liters of water.

You can pour water from one barrel to another. In one act of pouring, you can choose two different barrels $x$ and $y$ (the $x$-th barrel shouldn't be empty) and pour any possible amount of water from barrel $x$ to barrel $y$ (possibly, all water). You may assume that barrels have infinite capacity, so you can pour any amount of water in each of them. 

Calculate the maximum possible difference between the maximum and the minimum amount of water in the barrels, if you can pour water at most $k$ times.

Some examples:   if you have four barrels, each containing $5$ liters of water, and $k = 1$, you may pour $5$ liters from the second barrel into the fourth, so the amounts of water in the barrels are $[5, 0, 5, 10]$, and the difference between the maximum and the minimum is $10$;  if all barrels are empty, you can't make any operation, so the difference between the maximum and the minimum amount is still $0$. 


-----Input-----

The first line contains one integer $t$ ($1 \le t \le 1000$) — the number of test cases.

The first line of each test case contains two integers $n$ and $k$ ($1 \le k < n \le 2 \cdot 10^5$) — the number of barrels and the number of pourings you can make.

The second line contains $n$ integers $a_1, a_2, \dots, a_n$ ($0 \le a_i \le 10^{9}$), where $a_i$ is the initial amount of water the $i$-th barrel has.

It's guaranteed that the total sum of $n$ over test cases doesn't exceed $2 \cdot 10^5$.


-----Output-----

For each test case, print the maximum possible difference between the maximum and the minimum amount of water in the barrels, if you can pour water at most $k$ times.


-----Example-----
Input
2
4 1
5 5 5 5
3 2
0 0 0

Output
10
0""",
'keywords':[
    '[barrels]: Containers numbered from 1 to n, where the i-th barrel initially contains a_i liters of water. In the first example, there are 4 barrels, each containing 5 liters of water, represented as [5, 5, 5, 5].', 
    '[maximum difference]: The largest possible gap between the fullest and emptiest barrels after performing up to k pourings. For the first example, this value is 10, achieved by creating a barrel with 10 liters and another with 0 liters.'
]
},{'problem': """Mikhail walks on a Cartesian plane. He starts at the point $(0, 0)$, and in one move he can go to any of eight adjacent points. For example, if Mikhail is currently at the point $(0, 0)$, he can go to any of the following points in one move:   $(1, 0)$;  $(1, 1)$;  $(0, 1)$;  $(-1, 1)$;  $(-1, 0)$;  $(-1, -1)$;  $(0, -1)$;  $(1, -1)$. 

If Mikhail goes from the point $(x1, y1)$ to the point $(x2, y2)$ in one move, and $x1 \ne x2$ and $y1 \ne y2$, then such a move is called a diagonal move.

Mikhail has $q$ queries. For the $i$-th query Mikhail's target is to go to the point $(n_i, m_i)$ from the point $(0, 0)$ in exactly $k_i$ moves. Among all possible movements he want to choose one with the maximum number of diagonal moves. Your task is to find the maximum number of diagonal moves or find that it is impossible to go from the point $(0, 0)$ to the point $(n_i, m_i)$ in $k_i$ moves.

Note that Mikhail can visit any point any number of times (even the destination point!).


-----Input-----

The first line of the input contains one integer $q$ ($1 \le q \le 10^4$) — the number of queries.

Then $q$ lines follow. The $i$-th of these $q$ lines contains three integers $n_i$, $m_i$ and $k_i$ ($1 \le n_i, m_i, k_i \le 10^{18}$) — $x$-coordinate of the destination point of the query, $y$-coordinate of the destination point of the query and the number of moves in the query, correspondingly.


-----Output-----

Print $q$ integers. The $i$-th integer should be equal to -1 if Mikhail cannot go from the point $(0, 0)$ to the point $(n_i, m_i)$ in exactly $k_i$ moves described above. Otherwise the $i$-th integer should be equal to the the maximum number of diagonal moves among all possible movements.


-----Example-----
Input
3
2 2 3
4 3 7
10 1 9

Output
1
6
-1



-----Note-----

One of the possible answers to the first test case: $(0, 0) \to (1, 0) \to (1, 1) \to (2, 2)$.

One of the possible answers to the second test case: $(0, 0) \to (0, 1) \to (1, 2) \to (0, 3) \to (1, 4) \to (2, 3) \to (3, 2) \to (4, 3)$.

In the third test case Mikhail cannot reach the point $(10, 1)$ in 9 moves.""",
    'keywords': [
        "[revisiting]: The ability to pass through any point, including the destination, multiple times during the journey. In the second example (4, 3, 7), the optimal path includes revisiting coordinates: (0, 0) → (0, 1) → (1, 2) → (0, 3) → (1, 4) → (2, 3) → (3, 2) → (4, 3). This feature allows for maximizing diagonal moves even when the direct path wouldn't utilize all available moves."
    ]
},
]

humaneval_shot_1 = [
    {
        'problem':'''Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True''',
    'keywords':[
        "[closer to each other]: Describes the relationship between two numbers in the list when their absolute difference is less than the specified threshold. In the test case has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3), 2.8 and 3.0 are closer to each other than the threshold of 0.3, as their difference (0.2) is less than 0.3.",
        "[has_close_elements]: Function name that defines the operation to be implemented. It takes two arguments: a list of numbers and a threshold value. The function should return True if any two numbers in the list have a difference smaller than the threshold, and False otherwise."
        ]
    },{
       'problem':"""Input to this function is a string containing multiple groups of nested parentheses. Your goal is to separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']""",
   'keywords':[
"[balanced]: Each opening parenthesis '(' has a corresponding closing parenthesis ')' in the correct order. The test case shows all groups are balanced, e.g., '()' and '(())'.",
"[nested parentheses]: Groups of parentheses where inner pairs are completely contained within outer pairs, without overlapping. In the test case, '(()())' demonstrates this concept, with two complete inner pairs nested within an outer pair.",
"[separate_paren_groups]: Function name indicating the functionality to be implemented. This function takes a single string argument containing multiple groups of nested parentheses. It should return a list of separated, independent parentheses groups."
    ]
   }
]  



apps_shot_2=[{
    'problem':"""There are several cards arranged in a row, and each card has an associated number of points The points are given in the integer array cardPoints.
In one step, you can take one card from the beginning or from the end of the row. You have to take exactly k cards.
Your score is the sum of the points of the cards you have taken.
Given the integer array cardPoints and the integer k, return the maximum score you can obtain.

Example 1:
Input: cardPoints = [1,2,3,4,5,6,1], k = 3
Output: 12
Explanation: After the first step, your score will always be 1. However, choosing the rightmost card first will maximize your total score. The optimal strategy is to take the three cards on the right, giving a final score of 1 + 6 + 5 = 12.

Example 2:
Input: cardPoints = [2,2,2], k = 2
Output: 4
Explanation: Regardless of which two cards you take, your score will always be 4.

Example 3:
Input: cardPoints = [9,7,7,9,7,7,9], k = 7
Output: 55
Explanation: You have to take all the cards. Your score is the sum of points of all cards.

Example 4:
Input: cardPoints = [1,1000,1], k = 1
Output: 1
Explanation: You cannot take the card in the middle. Your best score is 1. 

Example 5:
Input: cardPoints = [1,79,80,1,1,1,200,1], k = 3
Output: 202


Constraints:

1 <= cardPoints.length <= 10^5
1 <= cardPoints[i] <= 10^4
1 <= k <= cardPoints.length
### Use Call-Based Format""",

    'response':'''class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        max_score = 0
        curr_score = 0
        init_hand = cardPoints[len(cardPoints)-k:]
        max_score = sum(init_hand)
        curr_score = max_score
        for i in range(k):
            curr_score -= init_hand[i]
            curr_score += cardPoints[i]
            if curr_score > max_score:
                max_score = curr_score
        return max_score'''
},{'problem':"""Your music player contains N different songs and she wants to listen to L (not necessarily different) songs during your trip.  You create a playlist so that:

Every song is played at least once
A song can only be played again only if K other songs have been played

Return the number of possible playlists.  As the answer can be very large, return it modulo 10^9 + 7.




Example 1:
Input: N = 3, L = 3, K = 1
Output: 6
Explanation: There are 6 possible playlists. [1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1].


Example 2:
Input: N = 2, L = 3, K = 0
Output: 6
Explanation: There are 6 possible playlists. [1, 1, 2], [1, 2, 1], [2, 1, 1], [2, 2, 1], [2, 1, 2], [1, 2, 2]


Example 3:
Input: N = 2, L = 3, K = 1
Output: 2
Explanation: There are 2 possible playlists. [1, 2, 1], [2, 1, 2]




Note:

0 <= K < N <= L <= 100
### Use Call-Based Format""",

    'response':'''import math

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        s = 0  
        c = 0  
        r = 0  
        x = math.factorial(N) 
        while True:
            c = x * ((N-r-K)**(L-N)) * (-1)**r // (math.factorial(N-r-K) * math.factorial(r))
            if c != 0:
                s = (s + c) % (10**9 + 7)
                r += 1
            else:
                return s'''}]



apps_shot_3 = [{
        'problem':"""You have $n$ barrels lined up in a row, numbered from left to right from one. Initially, the $i$-th barrel contains $a_i$ liters of water.

You can pour water from one barrel to another. In one act of pouring, you can choose two different barrels $x$ and $y$ (the $x$-th barrel shouldn't be empty) and pour any possible amount of water from barrel $x$ to barrel $y$ (possibly, all water). You may assume that barrels have infinite capacity, so you can pour any amount of water in each of them. 

Calculate the maximum possible difference between the maximum and the minimum amount of water in the barrels, if you can pour water at most $k$ times.

Some examples:   if you have four barrels, each containing $5$ liters of water, and $k = 1$, you may pour $5$ liters from the second barrel into the fourth, so the amounts of water in the barrels are $[5, 0, 5, 10]$, and the difference between the maximum and the minimum is $10$;  if all barrels are empty, you can't make any operation, so the difference between the maximum and the minimum amount is still $0$. 


-----Input-----

The first line contains one integer $t$ ($1 \le t \le 1000$) — the number of test cases.

The first line of each test case contains two integers $n$ and $k$ ($1 \le k < n \le 2 \cdot 10^5$) — the number of barrels and the number of pourings you can make.

The second line contains $n$ integers $a_1, a_2, \dots, a_n$ ($0 \le a_i \le 10^{9}$), where $a_i$ is the initial amount of water the $i$-th barrel has.

It's guaranteed that the total sum of $n$ over test cases doesn't exceed $2 \cdot 10^5$.


-----Output-----

For each test case, print the maximum possible difference between the maximum and the minimum amount of water in the barrels, if you can pour water at most $k$ times.


-----Example-----
Input
2
4 1
5 5 5 5
3 2
0 0 0

Output
10
0
### Use Standard Input Format (read inputs with `input()`, write results with `print()`)""",
    'response':"""\
for _ in range(int(input())):
    n = int(input())
    a = list(map(int, input().split()))
    ans, small, big = 0, 2*10**9, -1
    for i in range(len(a)-1): 
        if a[i] == -1 and a[i+1] != -1:
            small = min(small, a[i+1])
            big = max(big, a[i+1])
        if a[i] != -1 and a[i+1] == -1:
            small = min(small, a[i])
            big = max(big, a[i])
        if a[i] != -1 and a[i+1] != -1:
            ans = max(ans, abs(a[i]-a[i+1]))
    if big == -1: 
        print(ans, 0)
    else: 
        x = (small + big) // 2
        ans = max(ans, abs(big-x), abs(x-small))
        print(ans, x)"""
},{'problem':"""Mikhail walks on a Cartesian plane. He starts at the point $(0, 0)$, and in one move he can go to any of eight adjacent points. For example, if Mikhail is currently at the point $(0, 0)$, he can go to any of the following points in one move:   $(1, 0)$;  $(1, 1)$;  $(0, 1)$;  $(-1, 1)$;  $(-1, 0)$;  $(-1, -1)$;  $(0, -1)$;  $(1, -1)$. 

If Mikhail goes from the point $(x1, y1)$ to the point $(x2, y2)$ in one move, and $x1 \ne x2$ and $y1 \ne y2$, then such a move is called a diagonal move.

Mikhail has $q$ queries. For the $i$-th query Mikhail's target is to go to the point $(n_i, m_i)$ from the point $(0, 0)$ in exactly $k_i$ moves. Among all possible movements he want to choose one with the maximum number of diagonal moves. Your task is to find the maximum number of diagonal moves or find that it is impossible to go from the point $(0, 0)$ to the point $(n_i, m_i)$ in $k_i$ moves.

Note that Mikhail can visit any point any number of times (even the destination point!).


-----Input-----

The first line of the input contains one integer $q$ ($1 \le q \le 10^4$) — the number of queries.

Then $q$ lines follow. The $i$-th of these $q$ lines contains three integers $n_i$, $m_i$ and $k_i$ ($1 \le n_i, m_i, k_i \le 10^{18}$) — $x$-coordinate of the destination point of the query, $y$-coordinate of the destination point of the query and the number of moves in the query, correspondingly.


-----Output-----

Print $q$ integers. The $i$-th integer should be equal to -1 if Mikhail cannot go from the point $(0, 0)$ to the point $(n_i, m_i)$ in exactly $k_i$ moves described above. Otherwise the $i$-th integer should be equal to the the maximum number of diagonal moves among all possible movements.


-----Example-----
Input
3
2 2 3
4 3 7
10 1 9

Output
1
6
-1



-----Note-----

One of the possible answers to the first test case: $(0, 0) \to (1, 0) \to (1, 1) \to (2, 2)$.

One of the possible answers to the second test case: $(0, 0) \to (0, 1) \to (1, 2) \to (0, 3) \to (1, 4) \to (2, 3) \to (3, 2) \to (4, 3)$.

In the third test case Mikhail cannot reach the point $(10, 1)$ in 9 moves.
### Use Standard Input Format (read inputs with `input()`, write results with `print()`)""",
    'response':"""\
q = int(input())
for e in range(q):
    x, y, k = list(map(int, input().split()))
    x, y = abs(x), abs(y)
    x, y = max(x, y), min(x, y)
    if (x % 2 != k % 2):
        k -= 1
        y -= 1
    if (x > k):
        print(-1)
        continue
    if ((x - y) % 2):
        k -= 1
        x -= 1
    print(k)"""
}]

mbpp_shot_1 = [
    {
        'problem':"""Write a function to find the shared elements from the given two lists.
assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))""",
        'keywords':[
            '[shared elements]: Elements that appear in both input lists or sequences. In the test case, 4 and 5 are the shared elements between (3, 4, 5, 6) and (5, 7, 4, 10), as they occur in both sequences.',
            "[similar_elements]: Function name indicating the operation to be implemented. It takes two lists (or tuples) as input and should return a collection of elements common to both input sequences."
        ]
   }
]




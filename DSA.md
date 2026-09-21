Here is a comprehensive Markdown study guide for the coding interview questions listed in your images. It is structured in a LeetCode style, providing the intuition, complexity analysis, and clean C++ code for each problem.

---

Top Coding Interview Questions - C++ Solutions & Explanations

Part 1: Array Basics & Medium Problems

1. Check if the Array is Sorted II

Intuition:

· Iterate through the array and count how many times nums[i] > nums[i+1] (inversions).
· If there are 0 inversions, it's sorted.
· If there is exactly 1 inversion, check if rotating the array makes it sorted (i.e., nums[0] >= nums[n-1]).
· Otherwise, it's not sorted.

Complexity: Time: O(N), Space: O(1)

```cpp
bool check(vector<int>& nums) {
    int count = 0, n = nums.size();
    for (int i = 0; i < n; i++) {
        if (nums[i] > nums[(i + 1) % n]) count++;
    }
    return count <= 1;
}
```

2. Remove duplicates from Sorted array

Intuition:

· Use two pointers: i (slow pointer for unique elements) and j (fast pointer for iteration).
· If nums[j] != nums[i], increment i and update nums[i] = nums[j].
· Return i + 1 as the new length.

Complexity: Time: O(N), Space: O(1)

```cpp
int removeDuplicates(vector<int>& nums) {
    if (nums.empty()) return 0;
    int i = 0;
    for (int j = 1; j < nums.size(); j++) {
        if (nums[j] != nums[i]) {
            i++;
            nums[i] = nums[j];
        }
    }
    return i + 1;
}
```

3. Left Rotate Array by One

Intuition:

· Store the first element in a temporary variable.
· Shift all elements to the left by one position.
· Place the temporary variable at the last position.

Complexity: Time: O(N), Space: O(1)

```cpp
void rotateByOne(vector<int>& nums) {
    int temp = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        nums[i - 1] = nums[i];
    }
    nums[nums.size() - 1] = temp;
}
```

4. Left Rotate Array by K Places

Intuition:

· Reverse the first k elements.
· Reverse the remaining n-k elements.
· Reverse the entire array.
· (Alternatively, use std::rotate in C++).

Complexity: Time: O(N), Space: O(1)

```cpp
void rotate(vector<int>& nums, int k) {
    int n = nums.size();
    k = k % n;
    reverse(nums.begin(), nums.begin() + k);
    reverse(nums.begin() + k, nums.end());
    reverse(nums.begin(), nums.end());
}
```

5. Move Zeros to End

Intuition:

· Maintain a pointer j to track the position for the next non-zero element.
· Iterate through the array. If the current element is non-zero, swap it with nums[j] and increment j.

Complexity: Time: O(N), Space: O(1)

```cpp
void moveZeroes(vector<int>& nums) {
    int j = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] != 0) {
            swap(nums[i], nums[j]);
            j++;
        }
    }
}
```

6. Linear Search

Intuition:

· Iterate through the array and return the index if the target is found.
· Return -1 if the loop finishes without finding the target.

Complexity: Time: O(N), Space: O(1)

```cpp
int linearSearch(vector<int>& nums, int target) {
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] == target) return i;
    }
    return -1;
}
```

7. Union of two sorted arrays

Intuition:

· Use two pointers i and j for the two arrays.
· Compare elements. Add the smaller one to the result.
· Handle duplicates by skipping equal elements if they are already in the result.

Complexity: Time: O(N + M), Space: O(N + M) (for result)

```cpp
vector<int> findUnion(vector<int>& arr1, vector<int>& arr2) {
    int i = 0, j = 0;
    vector<int> result;
    while (i < arr1.size() && j < arr2.size()) {
        if (arr1[i] < arr2[j]) {
            if (result.empty() || result.back() != arr1[i]) result.push_back(arr1[i]);
            i++;
        } else if (arr1[i] > arr2[j]) {
            if (result.empty() || result.back() != arr2[j]) result.push_back(arr2[j]);
            j++;
        } else {
            if (result.empty() || result.back() != arr1[i]) result.push_back(arr1[i]);
            i++; j++;
        }
    }
    while (i < arr1.size()) { if (result.back() != arr1[i]) result.push_back(arr1[i]); i++; }
    while (j < arr2.size()) { if (result.back() != arr2[j]) result.push_back(arr2[j]); j++; }
    return result;
}
```

8. Find missing number

Intuition:

· Since the numbers are from 0 to N, use the XOR property (a ^ a = 0, a ^ 0 = a).
· XOR all array elements and XOR all numbers from 0 to N. The result is the missing number.

Complexity: Time: O(N), Space: O(1)

```cpp
int missingNumber(vector<int>& nums) {
    int xor1 = 0, xor2 = 0;
    for (int i = 0; i < nums.size(); i++) {
        xor2 ^= nums[i];
        xor1 ^= (i + 1);
    }
    return xor1 ^ xor2;
}
```

9. Maximum Consecutive Ones

Intuition:

· Keep a running count of consecutive 1s.
· If a 0 is encountered, reset the count to 0.
· Keep track of the maximum count seen so far.

Complexity: Time: O(N), Space: O(1)

```cpp
int findMaxConsecutiveOnes(vector<int>& nums) {
    int maxCount = 0, currentCount = 0;
    for (int num : nums) {
        if (num == 1) {
            currentCount++;
            maxCount = max(maxCount, currentCount);
        } else {
            currentCount = 0;
        }
    }
    return maxCount;
}
```

10. Find the number that appears once, and other numbers twice

Intuition:

· XOR all elements.
· Since a ^ a = 0, pairs will cancel each other out, leaving only the single number.

Complexity: Time: O(N), Space: O(1)

```cpp
int singleNumber(vector<int>& nums) {
    int result = 0;
    for (int num : nums) result ^= num;
    return result;
}
```

11. Longest subarray with given sum K (positives)

Intuition:

· Use the sliding window technique (two pointers).
· Expand the right pointer to increase the sum. If the sum exceeds K, shrink from the left.

Complexity: Time: O(N), Space: O(1)

```cpp
int longestSubarrayWithSumK(vector<int>& nums, long long k) {
    int left = 0, right = 0, maxLen = 0;
    long long sum = 0;
    while (right < nums.size()) {
        sum += nums[right];
        while (left <= right && sum > k) {
            sum -= nums[left];
            left++;
        }
        if (sum == k) maxLen = max(maxLen, right - left + 1);
        right++;
    }
    return maxLen;
}
```

12. Longest subarray with sum K (includes negatives)

Intuition:

· Sliding window fails with negatives. Use Prefix Sum + Hash Map.
· Store the first occurrence of each prefix sum.
· If currentSum - K exists in the map, we found a subarray.

Complexity: Time: O(N), Space: O(N)

```cpp
int longestSubarrayWithSumK(vector<int>& nums, int k) {
    unordered_map<long long, int> prefixSumMap;
    long long sum = 0;
    int maxLen = 0;
    for (int i = 0; i < nums.size(); i++) {
        sum += nums[i];
        if (sum == k) maxLen = i + 1;
        if (prefixSumMap.find(sum - k) != prefixSumMap.end()) {
            maxLen = max(maxLen, i - prefixSumMap[sum - k]);
        }
        if (prefixSumMap.find(sum) == prefixSumMap.end()) {
            prefixSumMap[sum] = i;
        }
    }
    return maxLen;
}
```

---

Part 2: String Problems

13. Remove Outermost Parentheses

Intuition:

· Keep a depth counter.
· Ignore the outermost ( (when depth > 0 before incrementing) and the outermost ) (when depth > 0 after decrementing).
· Append other characters to the result.

Complexity: Time: O(N), Space: O(N)

```cpp
string removeOuterParentheses(string s) {
    string result = "";
    int depth = 0;
    for (char c : s) {
        if (c == '(') {
            if (depth > 0) result += c;
            depth++;
        } else {
            depth--;
            if (depth > 0) result += c;
        }
    }
    return result;
}
```

14. Reverse words in a given string / Palindrome Check

Intuition:

· Reverse the entire string.
· Iterate through the reversed string and reverse each individual word.

Complexity: Time: O(N), Space: O(1) (if done in-place)

```cpp
string reverseWords(string s) {
    reverse(s.begin(), s.end());
    int n = s.size(), i = 0, l = 0, r = 0;
    while (i < n) {
        while (i < n && s[i] != ' ') s[r++] = s[i++];
        if (l < r) {
            reverse(s.begin() + l, s.begin() + r);
            s[r++] = ' ';
            l = r;
        }
        i++;
    }
    if (r > 0) s.resize(r - 1);
    return s;
}
```

15. Largest Odd Number in a String

Intuition:

· Iterate from the end of the string.
· The first odd digit encountered means the prefix up to that digit is the largest odd number.

Complexity: Time: O(N), Space: O(1)

```cpp
string largestOddNumber(string num) {
    for (int i = num.length() - 1; i >= 0; i--) {
        if ((num[i] - '0') % 2 != 0) {
            return num.substr(0, i + 1);
        }
    }
    return "";
}
```

16. Longest Common Prefix

Intuition:

· Sort the array of strings.
· The longest common prefix will be between the first and last string in the sorted array.
· Compare them character by character.

Complexity: Time: O(N log N * M) (sorting), Space: O(1)

```cpp
string longestCommonPrefix(vector<string>& strs) {
    if (strs.empty()) return "";
    sort(strs.begin(), strs.end());
    string first = strs[0], last = strs.back();
    int i = 0;
    while (i < first.size() && i < last.size() && first[i] == last[i]) i++;
    return first.substr(0, i);
}
```

17. Isomorphic String

Intuition:

· Use two hash maps (or arrays of size 256) to store the mapping from s to t and t to s.
· If a character is already mapped to a different character, return false.

Complexity: Time: O(N), Space: O(1) (constant space for fixed alphabet)

```cpp
bool isIsomorphic(string s, string t) {
    int m1[256] = {0}, m2[256] = {0};
    for (int i = 0; i < s.length(); i++) {
        if (m1[s[i]] != m2[t[i]]) return false;
        m1[s[i]] = i + 1;
        m2[t[i]] = i + 1;
    }
    return true;
}
```

18. Rotate String

Intuition:

· If s can be rotated to form goal, then goal must be a substring of s + s.
· Ensure lengths are equal.

Complexity: Time: O(N), Space: O(N)

```cpp
bool rotateString(string s, string goal) {
    if (s.length() != goal.length()) return false;
    string doubled = s + s;
    return doubled.find(goal) != string::npos;
}
```

19. Check if two strings are anagram of each other

Intuition:

· Use a frequency array of size 26.
· Increment for characters in s, decrement for characters in t.
· If all counts are 0, they are anagrams.

Complexity: Time: O(N), Space: O(1)

```cpp
bool isAnagram(string s, string t) {
    if (s.length() != t.length()) return false;
    int count[26] = {0};
    for (int i = 0; i < s.length(); i++) {
        count[s[i] - 'a']++;
        count[t[i] - 'a']--;
    }
    for (int i = 0; i < 26; i++) {
        if (count[i] != 0) return false;
    }
    return true;
}
```

---

Part 3: Array Advanced & Matrix Problems

20. Sort an array of 0's 1's and 2's (Dutch National Flag)

Intuition:

· Use three pointers: low, mid, high.
· mid traverses the array.
· If 0, swap with low, increment both.
· If 1, increment mid.
· If 2, swap with high, decrement high.

Complexity: Time: O(N), Space: O(1)

```cpp
void sortColors(vector<int>& nums) {
    int low = 0, mid = 0, high = nums.size() - 1;
    while (mid <= high) {
        if (nums[mid] == 0) {
            swap(nums[low], nums[mid]);
            low++; mid++;
        } else if (nums[mid] == 1) {
            mid++;
        } else {
            swap(nums[mid], nums[high]);
            high--;
        }
    }
}
```

21. Majority Element-I

Intuition:

· Use Boyer-Moore Voting Algorithm.
· Maintain a count and a candidate.
· If count is 0, set candidate to current element.
· If current element == candidate, count++, else count--.

Complexity: Time: O(N), Space: O(1)

```cpp
int majorityElement(vector<int>& nums) {
    int count = 0, candidate = 0;
    for (int num : nums) {
        if (count == 0) candidate = num;
        count += (num == candidate) ? 1 : -1;
    }
    return candidate;
}
```

22. Kadane's Algorithm (Maximum Subarray Sum)

Intuition:

· Maintain a currentSum and maxSum.
· At each step, currentSum = max(nums[i], currentSum + nums[i]).
· Update maxSum = max(maxSum, currentSum).

Complexity: Time: O(N), Space: O(1)

```cpp
int maxSubArray(vector<int>& nums) {
    int maxSum = INT_MIN, currentSum = 0;
    for (int num : nums) {
        currentSum += num;
        if (currentSum > maxSum) maxSum = currentSum;
        if (currentSum < 0) currentSum = 0;
    }
    return maxSum;
}
```

23. Print subarray with maximum subarray sum

Intuition:

· Extension of Kadane's. Keep track of start and end indices.
· When currentSum resets to 0, update the temporary start index.

Complexity: Time: O(N), Space: O(1)

```cpp
pair<int, vector<int>> maxSubArrayWithIndices(vector<int>& nums) {
    int maxSum = INT_MIN, currentSum = 0;
    int start = 0, end = 0, tempStart = 0;
    for (int i = 0; i < nums.size(); i++) {
        currentSum += nums[i];
        if (currentSum > maxSum) {
            maxSum = currentSum;
            start = tempStart;
            end = i;
        }
        if (currentSum < 0) {
            currentSum = 0;
            tempStart = i + 1;
        }
    }
    vector<int> subarray(nums.begin() + start, nums.begin() + end + 1);
    return {maxSum, subarray};
}
```

24. Stock Buy and Sell

Intuition:

· Keep track of the minimum price seen so far.
· Calculate profit if sold today. Update max profit.

Complexity: Time: O(N), Space: O(1)

```cpp
int maxProfit(vector<int>& prices) {
    int minPrice = INT_MAX, maxPro = 0;
    for (int price : prices) {
        minPrice = min(minPrice, price);
        maxPro = max(maxPro, price - minPrice);
    }
    return maxPro;
}
```

25. Rearrange array elements by sign

Intuition:

· Create a new array. Use two pointers: pos (0) and neg (1).
· Iterate through the original array. Place positives at pos and negatives at neg, incrementing by 2.

Complexity: Time: O(N), Space: O(N)

```cpp
vector<int> rearrangeArray(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n);
    int pos = 0, neg = 1;
    for (int num : nums) {
        if (num > 0) { result[pos] = num; pos += 2; }
        else { result[neg] = num; neg += 2; }
    }
    return result;
}
```

26. Next Permutation

Intuition:

· Find the pivot index i from the right where nums[i] < nums[i+1].
· Find the rightmost element greater than nums[i] and swap.
· Reverse the suffix starting at i+1.

Complexity: Time: O(N), Space: O(1)

```cpp
void nextPermutation(vector<int>& nums) {
    int n = nums.size(), i = n - 2;
    while (i >= 0 && nums[i] >= nums[i + 1]) i--;
    if (i >= 0) {
        int j = n - 1;
        while (nums[j] <= nums[i]) j--;
        swap(nums[i], nums[j]);
    }
    reverse(nums.begin() + i + 1, nums.end());
}
```

27. Leaders in an Array

Intuition:

· Iterate from the right, keeping track of the maximum element seen so far.
· If the current element is greater than the max, it's a leader.

Complexity: Time: O(N), Space: O(1) (excluding result)

```cpp
vector<int> leaders(vector<int>& nums) {
    vector<int> result;
    int maxFromRight = INT_MIN;
    for (int i = nums.size() - 1; i >= 0; i--) {
        if (nums[i] > maxFromRight) {
            result.push_back(nums[i]);
            maxFromRight = nums[i];
        }
    }
    reverse(result.begin(), result.end());
    return result;
}
```

28. Longest Consecutive Sequence in an Array

Intuition:

· Insert all elements into a hash set.
· Iterate through the set. If num - 1 is not in the set, it's the start of a sequence.
· Count consecutive elements.

Complexity: Time: O(N), Space: O(N)

```cpp
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> st(nums.begin(), nums.end());
    int maxLen = 0;
    for (int num : st) {
        if (st.find(num - 1) == st.end()) {
            int currentNum = num, currentLen = 1;
            while (st.find(currentNum + 1) != st.end()) {
                currentNum++; currentLen++;
            }
            maxLen = max(maxLen, currentLen);
        }
    }
    return maxLen;
}
```

29. Set Matrix Zeroes

Intuition:

· Use the first row and first column as markers.
· Iterate the matrix (excluding first row/col). If matrix[i][j] == 0, mark matrix[i][0] = 0 and matrix[0][j] = 0.
· Use a separate variable for the first column/row to avoid overwriting.

Complexity: Time: O(N*M), Space: O(1)

```cpp
void setZeroes(vector<vector<int>>& matrix) {
    int m = matrix.size(), n = matrix[0].size();
    bool firstRowZero = false, firstColZero = false;
    for (int i = 0; i < m; i++) if (matrix[i][0] == 0) firstColZero = true;
    for (int j = 0; j < n; j++) if (matrix[0][j] == 0) firstRowZero = true;
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (matrix[i][j] == 0) {
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }
    }
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (matrix[i][0] == 0 || matrix[0][j] == 0) matrix[i][j] = 0;
        }
    }
    if (firstRowZero) for (int j = 0; j < n; j++) matrix[0][j] = 0;
    if (firstColZero) for (int i = 0; i < m; i++) matrix[i][0] = 0;
}
```

30. Rotate matrix by 90 degrees

Intuition:

· Transpose the matrix (swap matrix[i][j] with matrix[j][i]).
· Reverse each row.

Complexity: Time: O(N^2), Space: O(1)

```cpp
void rotate(vector<vector<int>>& matrix) {
    int n = matrix.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            swap(matrix[i][j], matrix[j][i]);
        }
    }
    for (int i = 0; i < n; i++) {
        reverse(matrix[i].begin(), matrix[i].end());
    }
}
```

31. Print the matrix in spiral manner

Intuition:

· Maintain four pointers: top, bottom, left, right.
· Traverse right, down, left, up. Update boundaries after each direction.

Complexity: Time: O(N*M), Space: O(1)

```cpp
vector<int> spiralOrder(vector<vector<int>>& matrix) {
    vector<int> result;
    if (matrix.empty()) return result;
    int top = 0, bottom = matrix.size() - 1;
    int left = 0, right = matrix[0].size() - 1;
    while (top <= bottom && left <= right) {
        for (int i = left; i <= right; i++) result.push_back(matrix[top][i]);
        top++;
        for (int i = top; i <= bottom; i++) result.push_back(matrix[i][right]);
        right--;
        if (top <= bottom) {
            for (int i = right; i >= left; i--) result.push_back(matrix[bottom][i]);
            bottom--;
        }
        if (left <= right) {
            for (int i = bottom; i >= top; i--) result.push_back(matrix[i][left]);
            left++;
        }
    }
    return result;
}
```

32. Count subarrays with given sum

Intuition:

· Use Prefix Sum + Hash Map.
· Keep a running sum. If sum - k exists in the map, add its frequency to the count.
· Increment the frequency of the current sum in the map.

Complexity: Time: O(N), Space: O(N)

```cpp
int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> prefixSumMap;
    prefixSumMap[0] = 1;
    int sum = 0, count = 0;
    for (int num : nums) {
        sum += num;
        if (prefixSumMap.find(sum - k) != prefixSumMap.end()) {
            count += prefixSumMap[sum - k];
        }
        prefixSumMap[sum]++;
    }
    return count;
}
```

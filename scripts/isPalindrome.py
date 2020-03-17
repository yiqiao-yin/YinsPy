class Solution:
    def isPalindrome(array):
        string = array.lower()
        q = ""
        for i in string:
            if i.isalpha() or i.isnumeric():
                q = "".join([q,i])
        reversedString = ""
        for i in reversed(range(len(q))):
            reversedString += q[i]
        return q == reversedString

#print(Solution.isPalindrome('car, a rac'))
#print(Solution.isPalindrome('abcba'))
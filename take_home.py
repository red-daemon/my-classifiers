def exclamation_comprehension(string_list):
    res = [s+"!" for s in string_list]
    return res

def exclamation_recursion(string_list):
    if not string_list:
        return []
    return [string_list[0]+"!"] + exclamation_recursion(string_list[1:])

def main():
    string_l = ["a","b","c","d","e","f"]
    print(exclamation_recursion(string_l))
    print(exclamation_comprehension(string_l))

if __name__ == "__main__":
    main()
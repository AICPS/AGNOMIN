
# Function Name Tokenization

## 1. Introduction

This project is for extracting meaningful tokens from different function names. Many function names
follow naming conventions, such as `get_string_length()`. These kind of function names are easy to split. If we know the separator, tokens are easy to extract.
Some function names follow "camel" rules, e.g. `getStringLength()`. Although they do not have separators, we can split them using the capital letters.
However, for some funtion names which consist only lower-case letters, e.g. `getstringlength()`, traditional ways of tokenization usually fail. Another problem is that abbreviations are commonly used in function naming. A common example is "String" -> "Str".
It's more difficult to tokenize those function names. Our method uses several techniques to handle all cases mentioned above, and gives relatively good results.

## 2. Usage

To use this tool to tokenize function names, users should have a .txt file with each line should be a function name. All lines containing a function must be continuous, as the script will terminate tokenization upon reaching a newline without a function name.
Use the command `python tokenization.py [filepath]` to run the script, where `[filepath]` is the filepath of the .txt file described above. An abbreviation list is provided, called **"abbr.txt"**. It contains 209 frequently used abbreviations during coding. Users can add more abbreviations inside, but should follow the format. Two small dictionaries **"words_5k.txt"** and **words_1w.txt** are provided, which contain about 5000 frequently used words in English. The program will automatically takes in the txt file and process. Every 500 function names, it will pop a message on the terminal showing the current progress: e.g. `Current Process: 1500`. 
After it finishes, the program will generate two json files: `[filepath].json` and `[filepath]_mapping.json`. The former contains the  mapping of function name to list of tokens for every processed function name. The latter contains the mapping of substrings to their final tokens. Examples would include
"string" -> "string" (string is contained in words_5k and thus tokenizes to itself)
"str" -> "string" (str is a recognized abbreviation for string)
"nbr" -> "nb" (nbr is highly strongly to nb, which is contained in words_1w)

## 3. Techniques

There are two classes in `tokenization.py`. The first class `Utils` contains math related functions. The second
class `Function` is for splitting function names. Our program first follows traditional naming conventions to split the function names and get a string list for every function. For every word in the list,
if it exists in the dictionary, this word will be added to "word" list. Otherwise, the program will check abbreviation table. If it's an abbreviation, the full word will return and added to "abbr" list.
Remaining words will go through the similar matching process. The algorithm will calculate the similarity between this word and every word in the dictionary. If the similarity is larger than 0.6, it will be consider a potential result. After comparing all the
words, the one with highest similarity will be return as its full word, and added to "similar" list. If there are still words left, they will be in the "no_match" list.

### 3.1 Similar matching
The program will calculate a similarity score between two strings. The score consists three parts: forward similarity, backward similarity, and Levenshtein distance. Each part have different weight: 0.5 for forward similarity, 0.2 for backward similarity, and 0.3 for Levenshtein distance.
The reason is that when comparing two strings, letters in the front are more important than thos in the back. When changing only one letter, front one will have more influence that back one.
E.g. "strain" and "train"; "train" and "trains". When forward/backward similarity increase, their factors will also increase. vice versa. 
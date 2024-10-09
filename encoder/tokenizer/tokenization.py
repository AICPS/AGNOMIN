import math
import re
import sys
import json
import cProfile

class Utils :
    
    @staticmethod
    def  weight( x) :
        if (x == 0.5) :
            return x
        elif(x < 0.5) :
            return 2 * x * x
        else :
            return ((-2 * x * x) + 4 * x - 1)
    
    '''
    Calculates weights for forward and backward similarities.
    The weights always sum to 0.7.
    '''
    @staticmethod
    def  factor( forward,  backward) :
        f = 0.5
        b = 0.2
        base = 0.2
        if (forward == 0.5 and backward == 0.5) :
            return [forward, backward]
        elif(forward > 0.5) :
            f += (1 - ((1 - forward) * (1 - forward))) * base
            b = 0.7 - f
            return [f, b]
        elif(backward > 0.5) :
            b += float((math.pow(backward,5) * 0.5))
            f = 0.7 - b
            return [f, b]
        else :
            return [forward, backward]
    
    '''
    Calculates a weighted average of forward, backwards, and Levenshtein similarity.
    The factor method is called to calculate the weight of forwards and backwards similarity.
    Levenshtein similarity is always weighted at 0.3.
    '''
    @staticmethod
    def  similarity( str1,  str2) :
        forwardSimilarly = Utils.forwardSimilarly(str1, str2)
        backwardSimilarly = Utils.backwardSimilarly(str1, str2)
        levenshteinSimilarity = Utils.levenshtein(str1, str2)
        f_f = Utils.factor(forwardSimilarly, backwardSimilarly)[0]
        f_b = Utils.factor(forwardSimilarly, backwardSimilarly)[1]
        f_l = 0.3
        return f_f * Utils.weight(forwardSimilarly) + f_b * Utils.weight(backwardSimilarly) + f_l * levenshteinSimilarity
    
    @staticmethod
    def  forwardSimilarly( str1,  str2) :
        score = 0
        length = min(len(str1),len(str2))
        i = 0
        while (i < length) :
            if (str1[i] == str2[i]) :
                score += 1
            else :
                break
            i += 1
        similarity = (float(score)) / (float(max(len(str1),len(str2))))
        return similarity
    
    @staticmethod
    def  backwardSimilarly( str1,  str2) :
        score = 0
        length = min(len(str1),len(str2))
        biggerIndex = 0
        if (len(str1) >= len(str2)) :
            biggerIndex = len(str1) - len(str2)
            i = length - 1
            while (i >= 0) :
                if (str1[i + biggerIndex] == str2[i]) :
                    score += 1
                else :
                    break
                i -= 1
        else :
            biggerIndex = len(str2) - len(str1)
            i = length - 1
            while (i >= 0) :
                if (str1[i] == str2[i + biggerIndex]) :
                    score += 1
                else :
                    break
                i -= 1
        similarity = (float(score)) / (float(max(len(str1),len(str2))))
        return similarity
    
    @staticmethod
    def  levenshtein( str1,  str2) :
        len1 = len(str1)
        len2 = len(str2)
        dif = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        a = 0
        while (a <= len1) :
            dif[a][0] = a
            a += 1
        a = 0
        while (a <= len2) :
            dif[0][a] = a
            a += 1
        temp = 0
        i = 1
        while (i <= len1) :
            j = 1
            while (j <= len2) :
                if (str1[i - 1] == str2[j - 1]) :
                    temp = 0
                else :
                    temp = 1
                dif[i][j] = Utils.mininal(dif[i - 1][j - 1] + temp, dif[i][j - 1] + 1, dif[i - 1][j] + 1)
                j += 1
            i += 1
        # print("Diff steps: "+dif[len1][len2]);
        similarity = 1 - float(dif[len1][len2]) / max(len(str1),len(str2))
        # print("Levenshtein: "+similarity);
        return similarity
    
    @staticmethod
    def  mininal(*args) :
        return min(args)
    
class Function :
    name = None
    tokens = None
    tokens_word = None
    tokens_abbr = None
    tokens_similar = None
    tokens_no_match = None
    #FUNCTION_NAME_PATH = "function_name_test.txt"
    FILE_OUTPUT_SIMILAR = "function_similar.txt"
    FILE_OUTPUT_NO_MATCH = "function_no_match.txt"
    FILE_OUTPUT_WORD = "function_word.txt"
    mapping_dict = dict()

    def __init__(self, name) :
        self.name = name
        self.tokens =  []
        self.tokens_word =  []
        self.tokens_abbr =  dict()
        self.tokens_similar =  dict()
        self.tokens_no_match =  []
    def  getName(self) :
        return self.name
    def setName(self, name) :
        self.name = name
    def  getTokens(self) :
        return self.tokens
    def setTokens(self, tokens) :
        self.tokens = tokens
    def addTokens(self, token) :
        self.tokens.append(token)
    def addTokenList(self, tokenList) :
        self.tokens += tokenList
    def addTokenWord(self, token) :
        self.tokens_word.append(token)
    def addTokenWordList(self, tokenList) :
        self.tokens_word += tokenList
    def  toString(self) :
        str1 = "Function Name: \'" + self.name + "\'" + "\n"
        str1 += "   All Tokens: " + ", ".join(self.tokens) + "\n"
        str1 += "  Word Tokens: " + ", ".join(self.tokens_word) + "\n"
        str1 += "  Abbr Tokens: " + str(self.tokens_abbr) + "\n"
        str1 += "  Simi Tokens: " + str(self.tokens_similar) + "\n"
        str1 += " Other Tokens: " + ", ".join(self.tokens_no_match) + "\n"
        return str1
    
    @staticmethod
    def  startsWithDigitOrLetter( str1) :
        return re.compile("^[a-zA-Z0-9]").match(str1)
    
    @staticmethod
    def  startsWithUpperLetter( str1) :
        return re.compile("^[A-Z]").match(str1)
    
    @staticmethod
    def  splitOnNumberAndLetter(name) :
        regex = re.compile("[^a-zA-Z0-9]+")
        token_list = regex.split(name)
#         if (Function.startsWithDigitOrLetter(name) is None) :
#             token_list.pop(0)
        return token_list
        
    @staticmethod
    def  splitCamelCase( name) :
        token_list = None
        pattern = re.compile("[A-Z]+")
        name_tmp=re.sub(pattern,lambda x:"_"+x.group(0),name)
        pattern = re.compile("[A-Z][a-z]+")
        name_tmp=re.sub(pattern,lambda x:"_"+x.group(0),name_tmp)
        token_list = name_tmp.split("_")
        i = 0
        while (i < len(token_list)) :
            token_list[i] = token_list[i].lower()
            i += 1
        if (Function.startsWithUpperLetter(name) is not None) :
            token_list.pop(0)
        while '' in token_list:
            token_list.remove('')
        return token_list

    @staticmethod
    def  getMostMeaningfulSubstring( name,  dictionary,  abbrMap) :
        if (len(name) <= 2) :
            return name
        result = Function.getMostSimilarToken(dictionary, name, 1.0)
        compareResult = None
        while (len(name) >= 3) :
            name = name[0:len(name) - 1]
            if ((name in abbrMap.keys())) :
                return name
            compareResult = Function.getMostSimilarToken(dictionary, name, 1.0)
            if (float(compareResult[1]) > float(result[1])) :
                result = compareResult
                if ((float(compareResult[1])) == 1.0) :
                    break
        return result[0]

    @staticmethod
    def  splitAllSmallLetters( name,  dictionary,  abbrMap) :
        tokenList =  []
        if (len(name) <= 2) :
            tokenList.append(name)
            return tokenList
        frontIndex = 0
        result = None
        name += "**"
        fullName = name
        while (True) :
            result = Function.getMostMeaningfulSubstring(name, dictionary, abbrMap)
            if (result=="") :
                frontIndex += 1
                name = name[1:]
            else :
                if (frontIndex != 0) :
                    tokenList.append(fullName[0:frontIndex])
                    frontIndex = 0
                tokenList.append(result)
                name = name[len(result):]
                fullName = name
            if (len(name) <= 2) :
                if (frontIndex != 0) :
                    tokenList.append(fullName[0:frontIndex])
                break       
        tokenList = Function.removeRedundant(tokenList)
        tokenList = Function.removeSingleLetter(tokenList)
        return tokenList
    
    @staticmethod
    def  mergeList( list_1,  list_2) :
        return list_1 + list_2
    
    @staticmethod
    def  filter( token_list) :
        arrStr =  []
        pattern_1 = re.compile(r'[0-9]+')
        for token in token_list :
            if (token != "" and (pattern_1.match(token) is None) and (token not in arrStr)) :
                arrStr.append(token)
        return arrStr

    @staticmethod
    def  getDictionary( version) :
        if (not version=="5k" and not version=="1w") :
            raise Exception("IllegalArgumentException")
        dictionaryPath = "tokenizer/words_" + version + ".txt"
        reader = open(dictionaryPath,'r')
        list =  []
        line = None
        while (True) :
            line = reader.readline().strip('\n')
            if(line):
                list.append(line)
            else:
                break
        reader.close()
        return list

    @staticmethod
    def  getAbbreviations() :
        dictionaryPath = "tokenizer/abbr.txt"
        reader = open(dictionaryPath,'r')
        abbrMap =  dict()
        line = None
        word = None
        abbr = None
        pattern = re.compile("[a-zA-Z]+")
        while (True) :
            line = reader.readline()
            if(line == ""):
                break
            line = line.strip('\n')
            if (line != "") :
                word = line.lower()
                line = reader.readline().strip('\n')
                abbr = line.lower()
                abbrMap[abbr] = word
            else:
                continue
        reader.close()
        return abbrMap
    
    @staticmethod
    def  splitFunctionName(name) :
        token_list = None
        if (name.startswith("FUN_")) :
            return ["fun"]
        token_list = Function.splitOnNumberAndLetter(name)
        list_tmp1 = []
        list_tmp2 = None
        for str1 in token_list : #get_str_len -- > get str len
            list_tmp2 = Function.splitCamelCase(str1)
            list_tmp1 = Function.mergeList(list_tmp1, list_tmp2)
        token_list = Function.filter(list_tmp1)
        return token_list
    
    @staticmethod
    def  getMostSimilarToken( dictionary,  abbr,  base) :
        score = 0.0
        similar = ""
        for str1 in dictionary :
            score_tmp = Utils.similarity(str1, abbr)
            if (score_tmp >= base and score_tmp > score) :
                score = score_tmp
                similar = str1
        result = [similar, score]
        return result
    
    @staticmethod
    def  getFullTokenByAbbr( abbrMap,  token) :
        if ((token in abbrMap.keys())) :
            return abbrMap.get(token)
        else :
            raise Exception("RuntimeException")
    
    @staticmethod
    def  getFunctionList(function_name_path) :
        reader = open(function_name_path,'r')
        list =  []
        line = None
        functionName = None
        while (True):
            line = reader.readline().strip('\n')
            if(line):
                functionName = line.strip()
                list.append(functionName)
            else:
                break
        reader.close()
        return list
    
    @staticmethod
    def  processList( functionList) :
        dictionary = Function.getDictionary("5k")
        abbrMap = Function.getAbbreviations()
        functionItemList =  []
        if (functionList == None or len(functionList) == 0) :
            raise Exception("RuntimeException")
        counter = 0
        for name in functionList :
            counter += 1
            print(counter, ", ", name)
            split_list = Function.splitFunctionName(name)
            f = Function(name)
            f.addTokenList(split_list)
            for str1 in split_list :            
                if (str1 in Function.mapping_dict) :
                    f.addTokenWord(Function.mapping_dict[str1])
                else :
                    try :
                        full = Function.getFullTokenByAbbr(abbrMap, str1)
                        Function.mapping_dict[str1] = full
                        f.addTokenWord(full)
                    except Exception:
                        if (str1 in dictionary) : 
                            Function.mapping_dict[str1] = str1
                            f.addTokenWord(str1)
                        else :
                            dictionary = Function.getDictionary("1w")
                            similar = Function.getMostSimilarToken(dictionary, str1, 0.6)
                            if (similar[0]=="") :
                                dictionary = Function.getDictionary("5k")
                                smallLetterTokens = Function.splitAllSmallLetters(str1, dictionary, abbrMap)
                                for similarToken in smallLetterTokens :

                                    if (similarToken in Function.mapping_dict):
                                        f.addTokenWord(Function.mapping_dict[similarToken])
                                    elif (similarToken  in dictionary) :
                                        Function.mapping_dict[similarToken] = similarToken 
                                        f.addTokenWord(similarToken)
                                    else :
                                        try :
                                            full = Function.getFullTokenByAbbr(abbrMap, similarToken)
                                            Function.mapping_dict[similarToken] = full
                                            f.addTokenWord(full)
                                        except Exception as e1 :
                                            similarity = Function.getMostSimilarToken(dictionary, similarToken, 0.6)
                                            if (similar[0]=="") :
                                                Function.mapping_dict[similarToken] = similarToken
                                                f.addTokenWord(similarToken)
                                            else :
                                                Function.mapping_dict[similarToken] = similarity[0]
                                                f.addTokenWord(similarity[0])
                            else :
                                Function.mapping_dict[str1] = similar[0]
                                f.addTokenWord(similar[0])
            functionItemList.append(f)
            if (len(functionItemList) % 500 == 0) :
                print("Current Process: " + str(len(functionItemList)))
        return functionItemList
    
    @staticmethod
    def  removeRedundant( list1) :
        list2, seen = [], set()
        for str1 in list1:
            if not str1 in seen:
                seen.add(str1)
                list2.append(str1)
        return  list2
    
    @staticmethod
    def  removeSingleLetter( list1) :
        list2 = list(filter(lambda x : len(x) > 1, list1))
        return list2
    
    @staticmethod
    def main(function_name_path) :
        functionItemList = Function.processList(Function.getFunctionList(function_name_path))
        function_name_token_dict = {}
        for f in functionItemList :
            function_name_token_dict[f.name] = f.tokens_word
        with open((function_name_path +".json"), "w") as json_output:
            json.dump(function_name_token_dict, json_output)
        with open((function_name_path +"_mapping.json"), "w") as json_output:
            json.dump(Function.mapping_dict, json_output)
    

if __name__=="__main__":
    cProfile.run('Function.main(sys.argv[1])', 'profiling')
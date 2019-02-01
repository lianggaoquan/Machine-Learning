'''
MapReduce word counting example

'''

class doc(object):
    def __init__(self,name,content):
        self.name = name
        self.content = content
    
    def getName(self):
        return self.name
    
    def getContent(self):
        return self.content

doc1 = doc("doc1", "Hello world Bye world")
doc2 = doc("doc2", "Hello Hadoop good Bye Hadoop")
all_docs = [doc1, doc2]

def Map(K,V):
    '''
    K: document name
    V: document content
    '''
    output = []
    for w in V.split(" "):
        output.append({w:1})
    
    return output

def internal_grouping(outputs):
    dic = {}
    for o in outputs:
        for i in range(len(o)):
            word = list(o[i].keys())[0]
            if word not in dic:
                dic[word] = [1]
            else:
                dic[word].append(1)
    return dic


outputs = []
for d in all_docs:
    name = d.getName()
    content = d.getContent()
    output = Map(name,content)
    outputs.append(output)

dic = internal_grouping(outputs)
print(dic)
print("==================")

def Reduce(K,V):
    '''
    K: a word
    V: a list of counts
    '''
    count = 0
    for v in V:
        count += v
    dic[K] = count

for w in dic.keys():
    list_counts = dic[w]
    Reduce(w,list_counts)

print(dic)
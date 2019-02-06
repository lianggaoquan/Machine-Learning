from collections import Counter,defaultdict

def tokenize(document):
    return document.split(" ")

def word_count_old(documents):
    return Counter(word for document in documents for word in tokenize(document))

documents = ["i like milk", "i hate coffee", "i hate spark"]

print("old method:\n")
print(word_count_old(documents))
print("\n")

print("======================")

def wc_mapper(document):
    for word in tokenize(document):
        yield (word,1)

def wc_reducer(word,counts):
    yield (word,sum(counts))

def word_count(documents):
    collector = defaultdict(list)

    for document in documents:
        for word,count in wc_mapper(document):
            collector[word].append(count)
    
    return [output for word,counts in collector.items() for output in wc_reducer(word,counts)]

print("new method:\n")
print(word_count(documents))
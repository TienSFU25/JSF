import matplotlib.pyplot as plt
import numpy as np
import pdb

word_count = {}

with open('./data/snips/train/label', 'r') as f:
    text = f.readlines()

    for line in text:
        intent = line[:-1]
        
        if intent not in word_count:
            word_count[intent] = 0
        
        word_count[intent] += 1

print(word_count)

words = word_count.keys()
occurences = word_count.values()

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

# pdb.set_trace()
# ax.bar(words, [1,2,3,4,5,5,6])
# plt.bar(words, occurences, align='center', alpha=0.5)
fig, ax = plt.subplots()    
r1 = ax.bar(words, occurences, align='center', alpha=0.5)
plt.ylabel('Count')
plt.title('Intent count in SNIPS')
# plt.xticks(np.arange(len(words)), words)
# ax.set_yticks(np.arange(0, 81, 10))

autolabel(r1)

plt.show()

# import matplotlib.pyplot as plt; plt.rcdefaults()
# import numpy as np
# import matplotlib.pyplot as plt

# objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
# y_pos = np.arange(len(objects))
# performance = [10,8,6,4,2,1]

# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Usage')
# plt.title('Programming language usage')

# plt.show()
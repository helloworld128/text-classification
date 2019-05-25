# fp = open('data/sgns.sogounews.bigram-char', encoding='utf-8')
# cnt = -1
# for l in fp:
#     cnt += 1
#     line = l.split()
#     if len(line) != 301:
#         print(line[:3])
#         print(l[:20])
#         print(len(line))
# print(cnt)


fp = open('sina/sinanews.test', encoding='utf-8')
lineCount, wordCount = 0, 0
for l in fp:
    line = l.split()
    lineCount += 1
    wordCount += len(line) - 10
print(wordCount / lineCount)
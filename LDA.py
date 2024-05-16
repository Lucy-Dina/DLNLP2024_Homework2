import math
import jieba
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# The number of paragraphs extracted from each document
Paragraph = 63  # ceil (para_total / num_novel) = ceil (1000 / 16) = ceil (62.5) = 63
# The number of words contained in each extracted paragraph
Token = 500
# The number of cross validation
Cross_validation = 10
# The number of topics
TOPICS = 80
# Whether to use characters as the basic unit
Flag_chr = 1
svm_classifier = SVC(kernel='linear')

# Reading corpus content & Paragraph sampling
def read_novel(path):  # path : Only include the txt files of each novel, all other files are excluded
    content = []
    tags = []
    names = os.listdir(path)
    for name in names:
            con_seg = []
            novel_name = path + '\\' + name
            with open(novel_name, 'r', encoding='gbk', errors='ignore') as f:
                con = f.read()
                con = content_deal(con)
                # print("content_deal(con): ", con[:100])
                # Read stopwords & punctuation
                with open("cn_stopwords.txt", "r", encoding='utf-8') as fs:
                    stop_word = fs.read().split('\n')
                    fs.close()
                with open("cn_punctuation.txt", "r", encoding='utf-8') as fp:
                    punctuation_word = fp.read().split('\n')
                    fp.close()
                extra_characters = punctuation_word + stop_word
                # Segmentation
                con_seg = []
                if Flag_chr == 1:  # word segmentation by character
                    for character in con:
                        if (character not in extra_characters) and (not character.isspace()):  # No extra_characters & sapce
                            con_seg.append(character)
                    # con_seg += [char for char in con]
                else:  # Word segmentation by word
                    for word in jieba.lcut(con):
                        if (word not in extra_characters) and (not word.isspace()):  # No extra_characters & sapce
                            con_seg.append(word)
                # print("Segmented content sample:", con_seg[:100])
                con_list = list(con_seg)
                #  16 papers，select "Paragraph" "Token"-words paragraphs evenly from each article for modeling   ???
                #  pos : Number of words per section after document partitioning
                pos = int(len(con_list)//Paragraph)  #   "//" : Integer division, returns the integer part of the quotient (rounded down)
                for i in range(Paragraph):
                    con_sample = con_list[i * pos:i * pos + Token - 1]
                    content.append(con_sample)
                    tags.append(name)
                # for i in range(Paragraph):
                #     con_temp = con_temp + con_list[i*pos:i*pos+Token]
                # content.append(con_temp)
            f.close()
    return content, tags

# def read_novel_test(path):
#     content = []
#     names = os.listdir(path)
#     for name in names:
#             con_temp = []
#             novel_name = path + '\\' + name
#             with open(novel_name, 'r', encoding='ANSI') as f:
#                 con = f.read()
#                 con = content_deal(con)
#                 con = jieba.lcut(con)
#                 con_list = list(con)
#                 pos = int(len(con)//Paragraph)
#                 for i in range(Paragraph):
#                     # Remaining portion of "con_list" in "def read_novel"
#                     con_temp = con_temp + con_list[i*pos+(Token+1):i*pos+(2*Token)]
#                 content.append(con_temp)
#             f.close()
#     return content, names

# Preprocessing of corpus (sentence breaking & removing meaningless content)
def content_deal(content):
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    Non_CN = u'[^a-zA-Z0-9\u4e00-\u9fa5]'
    for a in ad:
        content = content.replace(a, '')
    content = re.sub(Non_CN, '', content)
    # content = ''.join(content.split())
    # print("Processed content length:", len(content))  # 打印处理后的长度
    return content

if __name__ == '__main__':
    log = open('log_Tokens{}_Topics{}_chr{}.txt'.format(Token, TOPICS, Flag_chr), "w")
    # mode='a':Set file permissions to read/write
    # ====== Data Import ======
    print("========= DATA IMPORT =========", file=log)
    # [data_txt, data_tags] = read_novel("E:\\Deep_NLP\\Homework2\\data_tmp")
    # [data_txt, files] = read_novel("D:\\Desktop\\zzzzzz\\2024_spring\\Deep_NLP\\Homework2\\data_tmp")
    # [data_txt, files] = read_novel("\\content\\drive\\MyDrive\\DLNLP\\jyxstxtqj_downcc.com")
    [data_txt, data_tags] = read_novel("E:\\Deep_NLP\\Homework2\\jyxstxtqj_downcc.com")
    # [data_txt, data_tags] = read_novel("D:\\Desktop\\zzzzzz\\2024_spring\\Deep_NLP\\Homework2\\jyxstxtqj_downcc.com")
    # print("data_txt[900]: ", data_txt[900])
    print("num_para_total: ", len(data_txt), file=log)

    # # 创建标签编码器
    # label_encoder = LabelEncoder()
    #
    # # 将文本标签转换为整数
    # encoded_labels = label_encoder.fit_transform(data_tags)

    kf = KFold(n_splits=Cross_validation)

    ECHO_CROSS = 0
    Accuracy = []
    Accuracy_svm = []
    svm_train_accuracies = []
    svm_test_accuracies = []

    for train, test in kf.split(data_txt):
        print("[[ECHO_CROSS]]: ", ECHO_CROSS, file=log)
        # print("train: ", train)
        # print("test: ", test)
        # tmp=train[882]
        # print("tmp", tmp)
        # print("data_txt: ", data_txt[tmp])
        train_txt = [data_txt[i] for i in train]
        test_txt = [data_txt[i] for i in test]
        # ====== MODEL TRAINING ======
        print("========= MODEL TRAINING =========", file=log)
        # ------Calculate Number & Frequency------
        # the topic of each word in each article coming from
        Topic_All = []
        # the number of words in each topic
        Topic_count = {}
        # The word frequency of each topic (16 novels in total)
        Topic_fre = {i: {} for i in range(TOPICS)}
        # Topic_fre0 = {};Topic_fre1 = {};Topic_fre2 = {};Topic_fre3 = {};
        # Topic_fre4 = {};Topic_fre5 = {};Topic_fre6 = {};Topic_fre7 = {};
        # Topic_fre8 = {};Topic_fre9 = {};Topic_fre10 = {};Topic_fre11 = {};
        # Topic_fre12 = {};Topic_fre13 = {};Topic_fre14 = {};Topic_fre15 = {}
        # the number of words in each article
        Doc_count = []
        # the number of words in each article for each topic
        Doc_fre = []
        i = 0
        for data in train_txt:
            # print("i: ", i)
            # print("data: ", data)
            topic = []
            docfre = {h: 0 for h in range(TOPICS)}  # initialize all "fre" of topics are 0
            for word in data:
                # Assign a random initial topic to each word
                topic_index = random.randint(0, TOPICS - 1)
                # print("topic_index: ", topic_index)
                topic.append(topic_index)
                if '\u4e00' <= word <= '\u9fa5':  # \u4e00~\u9fa5 : The scope of basic Chinese characters
                    # Count the total number of words for each topic
                    # print("Adding word:", word)
                    Topic_count[topic_index] = Topic_count.get(topic_index, 0) + 1
                    # ↑get(key,default value) : Obtain the corresponding value in the dictionary through the specified key.
                    # ↑If the key does not exist, return the specified default value; if not specified, return None
                    # ↓Count the word frequency of current article
                    docfre[topic_index] = docfre.get(topic_index, 0) + 1
                    # print("docfre: ", docfre)
                    # Count the word frequency of each topic
                    Topic_fre[topic_index][word] = Topic_fre[topic_index].get(word, 0) + 1
                    # exec('Topic_fre{}[word]=Topic_fre{}.get(word, 0) + 1'.format(topic_index, topic_index))  # execute
            Topic_All.append(topic)
            docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())  # dic.items() : Traverse key value pairs
            # print("docfre: ", docfre)
            # Sort all iterable objects
            # reverse=False : Ascending order
            # "key=lambda" : sorting basis is lambda function (sort according to custom rules, using parameter: key).
            # "x: x[0]" : x represents each element in the list, and x [0] returns the first element within the element using an index
            Doc_fre.append(docfre)  # the word frequency of all articles
            # print("DocFre: ", Doc_fre)
            # Count the total number of words in each article
            Doc_count.append(sum(docfre))
            # exec('print(len(Topic_fre{}))'.format(i))
            i += 1
        Topic_count = list(dict(sorted(Topic_count.items(), key=lambda x: x[0], reverse=False)).values())
        # Convert to array for convenient subsequent calculations
        Doc_fre = np.array(Doc_fre)  # Count from zero in each "for" loop above (corresponding to each line of novel text)
        Topic_count = np.array(Topic_count)  # Count from previous loop in each "for" loop above
        Doc_count = np.array(Doc_count)  # Similar to Doc_fre

        # print("Doc_fre: ", Doc_fre)
        # print("Topic_count: ", Topic_count)
        # print("Doc_count: ", Doc_count)

        # ------Calculate Probability & Iterative Updates------
        # The probability of each topic being selected in each text
        Doc_pro = []  # p(t|d)~θd
        # Record the new probability of each topic being selected after each iteration
        Doc_pronew = []
        for i in range(len(train_txt)):
            # Doc_fre[i] / Doc_count[i] = [nti] / n = [pti] = θd
            doc = np.divide(Doc_fre[i], Doc_count[i])
            Doc_pro.append(doc)
        Doc_pro = np.array(Doc_pro)
        # print("Doc_pro: ", Doc_pro)
        # Iteration stop flag
        stop = 0
        # Number of iteration
        loopcount = 1
        while stop == 0:
            i = 0  # different txt
            for data in train_txt:
                top = Topic_All[i]
                for w in range(len(data)):
                    word = data[w]
                    pro = []
                    topfre = []
                    if '\u4e00' <= word <= '\u9fa5':
                        for j in range(TOPICS):
                            # Read the frequency of the word appearing in each topic (equivalent to each txt)
                            topfre.append(Topic_fre[j].get(word, 0))
                            # exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))
                        # p(w|t)~φt=[pwi]=[Nwi]/N
                        # p(w|d)=p(w|t)*p(t|d)=p(t|d)*p(w|t)=Doc_pro*[Nwi]/N=Doc_pro[i]*topfre/Topic_count
                        pro = Doc_pro[i] * topfre / Topic_count
                        # Find the topic that MAX the product of the above probabilities
                        m = np.argmax(pro)  # new topic
                        # Update how many words are in each document for each topic
                        Doc_fre[i][top[w]] -= 1  # old topic
                        Doc_fre[i][m] += 1
                        # Update the total number of words for each topic
                        Topic_count[top[w]] -= 1
                        Topic_count[m] += 1
                        Topic_fre[top[w]][word] = Topic_fre[top[w]].get(word, 0) - 1
                        Topic_fre[m][word] = Topic_fre[m].get(word, 0) + 1
                        # exec('Topic_fre{}[word] = Topic_fre{}.get(word, 0) - 1'.format(top[w], top[w]))  # 更新每个topic该词的频数
                        # exec('Topic_fre{}[word] = Topic_fre{}.get(word, 0) + 1'.format(m, m))
                        top[w] = m
                Topic_All[i] = top
                i += 1
            # print("loop_train: ", loopcount)
            # print("Doc_fre_new: ", Doc_fre)
            # print("Topic_count_new: ", Topic_count)
            # Calculate the new probability of selecting each topic for each article
            if loopcount == 1:  # In the first iteration, assignment "Doc_pronew"
                for i in range(len(train_txt)):
                    doc = np.divide(Doc_fre[i], Doc_count[i])
                    Doc_pronew.append(doc)
                Doc_pronew = np.array(Doc_pronew)
            else:  # Starting from the second iteration, update "Doc_pronew"
                for i in range(len(train_txt)):
                    doc = np.divide(Doc_fre[i], Doc_count[i])
                    Doc_pronew[i] = doc
            # print("Doc_pro: ", Doc_pro)
            # print("Doc_pronew: ", Doc_pronew)
            # If the probability of selecting each topic in each article no longer changes, ...
            # ...it is considered that the model has been trained completely
            if (Doc_pronew == Doc_pro).all():
                stop = 1
            else:
                Doc_pro = Doc_pronew.copy()  # backups
            loopcount += 1
        # Output finally
        print('Final Output of Train:', file=log)
        print("Doc_pronew: ", Doc_pronew, file=log)  # The probability of selecting each topic for each article in the final training
        print("Number of loop_train: ", loopcount, file=log)  # number of iteration
        print('========= Model training completed！ =========', file=log)

        # ====== MODEL TESTING (similar to MODEL TRAINING) ======
        print("========= MODEL TESTING =========", file=log)
        # [test_txt, files] = read_novel_test("E:\\Deep_NLP\\Homework2\\data_tmp")
        # [test_txt, files] = read_novel_test("D:\\Desktop\\zzzzzz\\2024_spring\\Deep_NLP\\Homework2\\data_tmp")
        # [test_txt, files] = read_novel_test("D:\\Desktop\\zzzzzz\\2024_spring\\Deep_NLP\\Homework2\\jyxstxtqj_downcc.com")
        # the number of words  in each article
        Doc_count_test = []
        # the number of words in each article for each topic
        Doc_fre_test = []
        # the topic of each word in each article coming from
        Topic_All_test = []
        i = 0
        for data in test_txt:
            topic = []
            docfre = {h: 0 for h in range(TOPICS)}
            for word in data:
                topic_index = random.randint(0, TOPICS - 1)
                topic.append(topic_index)
                if '\u4e00' <= word <= '\u9fa5':
                    docfre[topic_index] = docfre.get(topic_index, 0) + 1
            Topic_All_test.append(topic)
            docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
            Doc_fre_test.append(docfre)
            Doc_count_test.append(sum(docfre))
            i += 1
        Doc_fre_test = np.array(Doc_fre_test)
        Doc_count_test = np.array(Doc_count_test)
        # print("Doc_fre_test: ", Doc_fre_test)
        # print("Doc_count_test: ", Doc_count_test)
        Doc_pro_test = []
        Doc_pronew_test = []
        for i in range(len(test_txt)):
            doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
            Doc_pro_test.append(doc)
        Doc_pro_test = np.array(Doc_pro_test)
        # print("Doc_pro_test: ", Doc_pro_test)
        stop = 0
        loopcount = 1
        while stop == 0:
            i = 0
            for data in test_txt:
                top = Topic_All_test[i]
                for w in range(len(data)):
                    word = data[w]
                    pro = []
                    topfre = []
                    if '\u4e00' <= word <= '\u9fa5':
                        for j in range(TOPICS):
                            topfre.append(Topic_fre[j].get(word, 0))
                            # exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))
                        pro = Doc_pro_test[i] * topfre / Topic_count
                        m = np.argmax(pro)
                        Doc_fre_test[i][top[w]] -= 1
                        Doc_fre_test[i][m] += 1
                        top[w] = m
                Topic_All_test[i] = top
                i += 1
            # print("loop_test: ", loopcount)
            # print("Doc_fre_test_new: ", Doc_fre_test)
            if loopcount == 1:
                for i in range(len(test_txt)):
                    doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
                    Doc_pronew_test.append(doc)
                Doc_pronew_test = np.array(Doc_pronew_test)
            else:
                for i in range(len(test_txt)):
                    doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
                    Doc_pronew_test[i] = doc
            # print("Doc_pro_test: ", Doc_pro_test)
            # print("Doc_pronew_test: ", Doc_pronew_test)
            if (Doc_pronew_test == Doc_pro_test).all():
                stop = 1
            else:
                Doc_pro_test = Doc_pronew_test.copy()
            loopcount += 1
        print('Final Output of Test:', file=log)
        print("Doc_pronew: ", Doc_pronew, file=log)
        print("Doc_pronew_test: ", Doc_pronew_test, file=log)
        print("Number of loop_test: ", loopcount, file=log)
        print('========= Model testing completed！ =========', file=log)
        # ====== Classification ======
        result_para = []  # Which one is the result in train_txt
        result_tag = []  # Which one is the result in data_tags
        round_tag = []  # real tag
        for k in range(len(test_txt)):
            pro = []
            for i in range(len(train_txt)):
                dis = 0
                for j in range(TOPICS):
                    # Calculate Euclidean distance
                    dis += (Doc_pro[i][j] - Doc_pro_test[k][j]) ** 2  # ** : ^
                pro.append(dis)
            m = pro.index(min(pro))
            m_data = train[m]  # index corresponding to data_txt in train
            result_para.append(m)
            result_tag.append(data_tags[m_data])
            r_data = test[k]  # index corresponding to data_txt in test
            round_tag.append(data_tags[r_data])
            # print("pro: ", pro)
        print("========= Result of Classification =========", file=log)
        # print("result_para: ", result_para, file=log)
        # print("result_tag: ", result_tag, file=log)
        # print("round_tag: ", round_tag, file=log)
        # Compare & Calculate Accuracy
        correct_count = 0
        total_count = len(round_tag)
        for predicted, actual in zip(result_tag, round_tag):
            if predicted == actual:
                correct_count += 1
        acc = correct_count / total_count
        print("Correct number of test_txt: ", correct_count, file=log)
        print("Total number of test_txt: ", total_count, file=log)
        print("Accuracy in ECHO_CROSS {}: {:.2f}%".format(ECHO_CROSS, acc * 100), file=log)
        Accuracy.append(acc)

        X_train, X_test = Doc_pronew, Doc_pronew_test
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform([data_tags[hh] for hh in train])
        y_test = label_encoder.fit_transform([data_tags[hh] for hh in test])
        # y_train = [data_tags[hh] for hh in train]
        # y_test = [data_tags[hh] for hh in test]

        svm_classifier.fit(X_train, y_train)
        acc_tmp = svm_classifier.score(X_test, y_test)
        Accuracy_svm.append(acc_tmp)
        train_accuracy = svm_classifier.score(X_train, y_train)
        test_accuracy = svm_classifier.score(X_test, y_test)
        svm_train_accuracies.append(train_accuracy)
        svm_test_accuracies.append(test_accuracy)

        # predictions = svm_classifier.predict(X_test)
        # acc_tmp = accuracy_score(y_test, predictions)
        # Accuracy_svm.append(acc_tmp)
        print("Accuracy_SVM in ECHO_CROSS {}: {:.2f}%".format(ECHO_CROSS, acc_tmp * 100), file=log)
        ECHO_CROSS += 1

    print("The Whole Accuracy: ", Accuracy, file=log)
    print("The MEAN of Accuracy：", np.mean(Accuracy), file=log)
    print("The Whole Accuracy_SVM: ", Accuracy_svm, file=log)
    print("The MEAN of Accuracy_SVM：", np.mean(Accuracy_svm), file=log)
    print("The MEAN of Accuracy_SVM of train_txt：", np.mean(svm_train_accuracies), file=log)
    log.close()
    # figure
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.plot(np.arange(1, len(Accuracy) + 1), Accuracy, 'g', label='Accuracy_Euclidean')
    plt.plot(np.arange(1, len(svm_train_accuracies) + 1), svm_train_accuracies, 'b', label='Accuracy_SVM of train_txt')
    plt.plot(np.arange(1, len(svm_test_accuracies) + 1), svm_test_accuracies, 'r', label='Accuracy_SVM of test_txt')
    plt.xlabel('Echo_Cross')
    plt.ylabel('Accuracy')
    plt.title('Accuracy_Tokens{}_Topics{}_chr{}'.format(Token, TOPICS, Flag_chr))
    plt.legend()
    plt.grid(True)
    # plt.savefig('Training and Test Accuracy_word_T120.png')
    plt.savefig('Accuracy_Tokens{}_Topics{}_chr{}.png'.format(Token, TOPICS, Flag_chr))
    plt.show()







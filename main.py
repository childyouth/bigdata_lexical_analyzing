import re
from konlpy.tag import Kkma
from konlpy.tag import Mecab
from konlpy.tag import Hannanum
from konlpy.tag import Okt
from multiprocessing import Process, Queue
import gensim
import numpy as np
import time

konlpy = {"Kkma":Kkma,"Mecab":Mecab,"Okt":Okt,"Hannanum":Hannanum}

def make_dictionary(sentence):
    cnt = {}
    for c in sentence:  # 문장의 최소단위 (음절은 한글자. 어절은 공백단위와 특수문자)
        if c in cnt.keys():
            cnt[c] += 1
        else:
            cnt[c] = 1
    return cnt

def sort_result(result, N):
    if len(result) > N:
        result = sorted(result, key= lambda sentence : sentence[0], reverse=True)[:N]

    return set(result)

def cos_sim(A,B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def find_equal_by_cos(word2vec, lex_name,N,s1,s2,corpus):
    start_time = time.time()

    result = set()

    A = np.zeros(200)
    for i in s1:
        if i in word2vec:
            A = A + word2vec[i]

    i = 0
    for B_ in s2:
        if i % 50000 == 0:
            print(lex_name + " (word_embed): 현재 {} 개 비교".format(i))
        original_B = corpus[i]
        i+=1

        B = np.zeros(200)
        for b in B_:
            if b in word2vec:
                B = B + word2vec[b]

        similar_rate = cos_sim(A,B)
        result.add((similar_rate,original_B))
        result = sort_result(result,N)

    print(lex_name + " : 비교 완료")
    end_time = time.time()
    total_time = end_time - start_time

    return total_time,result



def find_equal(lex_name, N, s1, s2, corpus):
    start_time = time.time()

    result = set()
    s1 = make_dictionary(s1)
    i = 0
    for s2_cnt in s2:
        if i % 50000 == 0:
            print(lex_name + " : 현재 {} 개 비교".format(i))

        original_s2 = corpus[i]
        i += 1
        s1_cnt = s1
        s2_cnt = make_dictionary(s2_cnt)

        if sum(s1_cnt.values()) > sum(s2_cnt.values()):
            s1_cnt,s2_cnt = s2_cnt,s1_cnt   # s1_cnt의 형태소의 총개수가 항상 더 작다

        cnt = 0     # 공통음절 cnt

        for k in s1_cnt.keys():
            if k in s2_cnt.keys():
                cnt += s1_cnt[k] > s2_cnt[k] and s2_cnt[k] or s1_cnt[k]

        similar_rate = cnt / sum(s1_cnt.values())
        result.add((similar_rate, original_s2))
        result = sort_result(result, N)

    print(lex_name + " : 비교 완료")
    end_time = time.time()
    total_time = end_time - start_time

    return total_time, result



def extract_morphs(lexical_analyzer, lex_name, corpus, s1):
    start_time = time.time()

    s1 = lexical_analyzer.morphs(s1)
    s2 = list()

    i = 0
    for line in corpus:
        if i % 50000 == 0:
            print(lex_name + " : 현재 형태소 추출 {} 개".format(i))
        i+=1
        if not line: break
        try:
            s2.append(lexical_analyzer.morphs(line))
        except Exception as err:       # 일부 문장에서 단어중 추출이 안되는 경우가 있음. 그 문장은 비교에서 배제
            print(lex_name + " : 분석 불가 문장 - " + line)
            continue
    print(lex_name + " : 형태소 추출 완료")
    end_time = time.time()
    total_time = end_time - start_time

    return total_time, s1, s2

def run_calc(lex_name, corpus, sentence, N, w2v, lex, message):  # w2v : 0 = 사용안함 | 1 = 이미 train한 가중치 사용 | 2 = 지금 train하기
    result_info = ""
    lexical_anayzer = konlpy[lex_name]()
    time1 = 0
    s1 = sentence
    s2 = corpus
    start_time = time.time()
    if lex > 0:
        try:
            tokenized_file = open(lex_name+"_tokenized.txt","r")
            tmp = tokenized_file.read().strip().split('\n')
            tokenized_file.close()
        except FileNotFoundError as err:
            lex = 0
            print("형태소 추출 파일이 없습니다. 추출을 실행합니다.")
        else:
            if len(tmp) < lex * 10000:
                lex = 0
                print("형태소 추출양이 부족합니다. 추출을 실행합니다.")
            else:
                s1 = lexical_anayzer.morphs(s1)
                s2 = [i.split(' ') for i in tmp]

    if lex == 0:
        time1, s1, s2 = extract_morphs(lexical_anayzer, lex_name, corpus, sentence)
        result_info = result_info + "형태소 추출 걸린 시간 : {:.2f} 초\n".format(time1)

        with open(lex_name + "_tokenized.txt","w") as f:
            for i in s2:
                f.write(" ".join(i) + "\n")



    if w2v == 0:
        time2, results = find_equal(lex_name, N, s1, s2, corpus)

    elif w2v == 1:
        try:
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(lex_name + "_word2vec_weight")
            time2, results = find_equal_by_cos(word2vec, lex_name, N, s1,s2, corpus)
        except FileNotFoundError as err:
            w2v = 2
            print(lex_name + " : 사전 학습된 w2v 가 없습니다.")

    if w2v == 2:
        word_embed_time1 = time.time()
        print(lex_name + " : 학습을 시작합니다.")

        word2vec = gensim.models.Word2Vec(sentences=s2, size=200, window=3, min_count=2, workers=4,
                                          sg=0)  # sg=1 과 속도비교 해보기
        word2vec.wv.save_word2vec_format(lex_name + "_word2vec_weight")
        print(lex_name + " : 학습을 완료했습니다.")
        word_embed_time = time.time() - word_embed_time1
        time2, results = find_equal_by_cos(word2vec,lex_name,N,s1,s2, corpus)



    results = sorted(results, key=lambda sentence: sentence[0], reverse=True)
    end_time = time.time()

    if w2v == 2:
        result_info = result_info + "워드 임베딩 학습에 걸린 시간 : {:.2f} 초\n".format(word_embed_time)

    result_info = result_info + "순위 비교 걸린 시간 : {:.2f} 초\n".format(time2)
    result_info = result_info + "프로세스 완료까지 걸린시간 : {:.2f}초".format(end_time - start_time)

    result_type = "형태소 분석기 : " + lex_name
    message.put([result_type,results,result_info])

def MAIN():
    f_in = open("input.txt", "r",encoding='UTF8')
    print("말뭉치를 가져오는 중입니다...")
    corpus = f_in.read().strip().split('\n')[:-1]
    print("완료..!")
    f_in.close()

    s1 = (input("비교할 문장 :")).strip()
    N = int(input("표시할 유사문장 갯수 :"))
    corp_len = int(input("말뭉치 범위 ( 총 4억개의 문장. 만단위 입력 ) : "))
    corpus = corpus[:corp_len*10000]
    lex = int(input("이미 분석된 형태소 사용 ( 0 - 새로 분석 , 사용할 길이(말뭉치 범위와 동일) ) : "))
    w2v = int(input("Word Embeding ? ( 0 - no , 1 - pre_trained , 2 - training ) : "))
    #
    # word2vec = gensim.models.Word2Vec.load('ko.bin')

    # a = word2vec.wv.most_similar("강아지")
    # print(a)

    result = Queue()

    #ps_kkma = Process(target=run_calc, args=("Kkma",corpus,s1,N,w2v, lex,result))   # 너무 느려서 사용하지 않을것
    ps_mecab = Process(target=run_calc, args=("Mecab",corpus,s1,N,w2v, lex,result))
    ps_okt = Process(target=run_calc, args=("Okt",corpus,s1,N,w2v, lex,result))
    ps_hannanum = Process(target=run_calc, args=("Hannanum",corpus,s1,N,w2v, lex,result))

    main_t1 = time.time()

    #ps_kkma.start()
    ps_mecab.start()
    ps_okt.start()
    ps_hannanum.start()

    #ps_kkma.join()
    ps_mecab.join()
    ps_okt.join()
    ps_hannanum.join()

    print("\n\n모든 프로세스가 종료된 시점 : {:.2f}초".format(time.time()-main_t1),end="\n\n")

    result.put(None)
    while True:
        r = result.get()
        if r is None:
            break
        else:
            print("#"*20)
            print()
            print(r[0]+"\n")
            for i in r[1]:
                print(i)
            print(r[2] + "\n\n")
            print("#"*20)


if __name__ == "__main__":
    MAIN()

    try:
        print()
    except Exception:
        print()
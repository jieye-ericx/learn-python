from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np

# 特征抽取
#
# 导入包
from sklearn.feature_extraction.text import CountVectorizer

#
# # 实例化CountVectorizer
#
vector = CountVectorizer()
#
# # 调用fit_transform输入并转换数据
#
res = vector.fit_transform(["life is short,i like python", "life is too long,i dislike python"])


#
# # 打印结果
# print(vector.get_feature_names())
#
# print(res.toarray())


def dictvec():
    """
    字典数据抽取
    :return: None
    """
    # 实例化
    # dict = DictVectorizer(sparse=False)
    #
    # # 调用fit_transform
    # data = dict.fit_transform([{'city': '北京','temperature': 100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature': 30}])
    # #
    # #
    # dict.get_feature_names()
    # print(dict.inverse_transform(data))
    #
    # print(data)
    onehot = DictVectorizer()  # 如果结果不用toarray，请开启sparse=False
    instances = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60},
                 {'city': '深圳', 'temperature': 30}]
    X = onehot.fit_transform(instances).toarray()
    print(onehot.inverse_transform(X))
    print(onehot.get_feature_names())
    print(X)

    return None


def countvec():
    """
    对文本进行特征值化
    :return: None
    """
    # content = ["life is short,i like python", "life is too long,i dislike python"]
    # vectorizer = CountVectorizer()
    # print(vectorizer.fit_transform(content).toarray())
    cv = CountVectorizer()

    data = cv.fit_transform(["人生 苦短，我 喜欢 python", "人生漫长，不用 python"])

    print(cv.get_feature_names())  # 统计所有词，重复的只算一次的列表
    print(data)  # 输出的是一种sparse形式，节省资源
    print(data.toarray())  # 对每段str统计每个词出现的次数，单个字母不统计

    return None


def cutword():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 吧列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    print(c1)
    print(c2)
    print(c3)
    return c1, c2, c3


def hanzivec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)

    cv = CountVectorizer()

    data = cv.fit_transform([c1, c2, c3])

    print(cv.get_feature_names())

    print(data.toarray())

    return None


def tfidfvec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names())

    print(data.toarray())

    return None


def mm():
    """
    归一化处理
    :return: NOne
    """
    mm = MinMaxScaler(feature_range=(2, 3))

    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])

    print(data)

    return None


def stand():
    """
    标准化缩放
    :return:
    """
    std = StandardScaler()

    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])

    print(data)

    return None


def im():
    """
    缺失值处理
    :return:NOne
    """
    # NaN, nan
    im = SimpleImputer(missing_values=np.nan, strategy='mean')

    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])  # 按照列取了平均值

    print(data)

    return None


def var():
    """
    特征选择-删除低方差的特征
    :return: None
    """
    var = VarianceThreshold(threshold=2.0)

    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])

    print(data)
    return None


def pca():
    """
    主成分分析进行特征降维
    :return: None
    """
    pca = PCA(n_components=0.9)
    """
    小数 90%-95%
    整数 减小到的特征数量
    """

    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])

    print(data)

    return None


if __name__ == "__main__":
    # dictvec()
    # countvec()
    cutword()
    # hanzivec()
    # tfidfvec()
    # mm()
    # im()
    # var()
    # pca()
    # stand()

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def mylinear():
    lb = load_boston()
    # print(lb.DESCR)
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    # print(x_train)
    # print(x_test)
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.fit_transform(x_test)

    std_y=StandardScaler()

    y_train = std_y.fit_transform(y_train.reshape(-1,1))
    y_test = std_y.fit_transform(y_test.reshape(-1, 1))

    lr= LinearRegression()
    lr.fit(x_train,y_train)
    print(lr.coef_)

    y_predict=std_y.inverse_transform(lr.predict(x_test))
    print('ceshijiage',y_predict)

    return None


if __name__ == "__main__":
    mylinear()
#
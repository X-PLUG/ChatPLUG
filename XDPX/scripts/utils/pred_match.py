import os


class Model:
    def __init__(self, sess, path):
        from scripts.serve import Model as XDPXModel
        cwd = os.getcwd()
        os.chdir(path)
        self.model = XDPXModel(path, checkpoint=os.path.join(path, 'checkpoint.pt'))
        os.chdir(cwd)

    def predict(self, sess, texts):
        _, probs = self.model.predict([[' '.join(line[0]), ' '.join(line[1])] for line in texts])
        return [p[1] for p in probs]


def main():
    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models/timi-qq')

    sess = None
    model = Model(sess, export_path)
    print(model.predict(sess, [
        ['充满 电能 用 多久'.split(), '能用 多久 ， 可以 用 多少 时间'.split()],
        ['发 什么 快递'.split(), '刚 发 了 快递'.split()]
    ]))


if __name__ == "__main__":
    main()

import os


class Model:
    def __init__(self, sess, path):
        from scripts.serve import Model as XDPXModel
        cwd = os.getcwd()
        os.chdir(path)
        self.model = XDPXModel(path, checkpoint='checkpoint.pt')
        os.chdir(cwd)

    def predict(self, sess, texts):
        return self.model.predict([[' '.join(text)] for text in texts])[0]


def main():
    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models/cainiao-cls')

    sess = None
    model = Model(sess, export_path)
    print(model.predict(sess, ['ems 上门取件 多少钱'.split(), '今天 8点 之前 能 送到 不'.split()]))
    print(model.predict(sess, ['ems 上门取件 多少钱'.split()]))


if __name__ == "__main__":
    main()

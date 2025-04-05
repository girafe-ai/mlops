def test_imports():
    from cats_and_dogs.pl_modules.classifiers import ConvClassifier, SimpleClassifier
    from cats_and_dogs.pl_modules.data import MyDataModule
    from cats_and_dogs.pl_modules.model import ImageClassifier

    ConvClassifier, SimpleClassifier, MyDataModule, ImageClassifier
    assert True, "Problems with imports!"

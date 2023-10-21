import mlflow.pyfunc


class PyModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        return model_input["a"] + model_input["b"]


def main():
    mlflow.pyfunc.save_model(path="model_dir", python_model=PyModelWrapper())


if __name__ == "__main__":
    main()
